#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

/* Replay the first tokens of a retained C case through a shared libds4.
 *
 * Export prompt and expected IDs from case.json first:
 *
 *   python3 -c 'import json, sys; c = json.load(open(sys.argv[1])); \
 * print("\n".join(str(t) for t in c["prompt"] + c["sampled"]))' \
 *     case.json > case.tokens
 *
 * Build against the same native source and shared library used by ds4-py:
 *
 *   cc -O2 -I/workspace/ds4 bench/c_reference_driver.c \
 *     -L/workspace/native -Wl,-rpath,/workspace/native -lds4 \
 *     -o /tmp/c-replay-diagnostic
 *
 * Then run with the case's prompt length and original cached-prefix length.
 * Native logs stay on stderr; stdout is NDJSON diagnostic evidence.
 */

#include "ds4.h"
#include "c_sha256.h"

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TOP_COUNT 12
#define Q8_F16_CACHE_MB "5120"

typedef struct {
    const char *model;
    const char *token_file;
    const char *cases_file;
    const char *source_trace;
    const char *backend_name;
    const char *think_name;
    ds4_backend backend;
    int ctx;
    int prompt_tokens;
    int split_at;
    int steps;
} replay_config;

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "usage: %s --model MODEL --tokens FILE --prompt-tokens N [options]\n"
            "       %s --model MODEL --cases FILE --source-trace TRACE [options]\n"
            "  --backend cuda|cpu   native backend (default: cuda)\n"
            "  --ctx N              context size (default: 100000)\n"
            "  --split-at N         prefill N tokens before the full prompt\n"
            "  --steps N            tokens to capture (default: 16)\n"
            "  --think LEVEL        native think level (default: high)\n"
            "  --cases FILE         TSV case manifest (manifest mode)\n"
            "  --source-trace TRACE retained source trace for manifest evidence\n",
            program,
            program);
}

static int parse_number(const char *text, const char *option, int *out) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!text[0] || !end || *end || value < 0 || value > 1000000) {
        fprintf(stderr, "invalid %s value: %s\n", option, text);
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_options(int argc, char **argv, replay_config *cfg) {
    *cfg = (replay_config){
        .backend_name = "cuda",
        .backend = DS4_BACKEND_CUDA,
        .think_name = "high",
        .ctx = 100000,
        .prompt_tokens = -1,
        .steps = 16,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = i + 1 < argc ? argv[i + 1] : NULL;
        if (!value) {
            fprintf(stderr, "%s requires a value\n", arg);
            return 0;
        }
        if (!strcmp(arg, "--model")) cfg->model = value;
        else if (!strcmp(arg, "--tokens")) cfg->token_file = value;
        else if (!strcmp(arg, "--cases")) cfg->cases_file = value;
        else if (!strcmp(arg, "--source-trace")) cfg->source_trace = value;
        else if (!strcmp(arg, "--prompt-tokens")) {
            if (!parse_number(value, arg, &cfg->prompt_tokens)) return 0;
        } else if (!strcmp(arg, "--ctx")) {
            if (!parse_number(value, arg, &cfg->ctx)) return 0;
        } else if (!strcmp(arg, "--split-at")) {
            if (!parse_number(value, arg, &cfg->split_at)) return 0;
        } else if (!strcmp(arg, "--steps")) {
            if (!parse_number(value, arg, &cfg->steps)) return 0;
        } else if (!strcmp(arg, "--think")) cfg->think_name = value;
        else if (!strcmp(arg, "--backend")) {
            cfg->backend_name = value;
            if (!strcmp(value, "cuda")) cfg->backend = DS4_BACKEND_CUDA;
            else if (!strcmp(value, "cpu")) cfg->backend = DS4_BACKEND_CPU;
            else {
                fprintf(stderr, "unsupported backend: %s\n", value);
                return 0;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", arg);
            return 0;
        }
        i++;
    }
    if (!cfg->model || cfg->ctx <= 0) return 0;
    if (cfg->cases_file) {
        return cfg->source_trace && cfg->source_trace[0] && !cfg->token_file &&
               cfg->prompt_tokens < 0 && cfg->split_at == 0 && cfg->steps == 16;
    }
    return !cfg->source_trace && cfg->token_file && cfg->prompt_tokens > 0 && cfg->steps > 0 &&
           cfg->split_at <= cfg->prompt_tokens;
}

static int load_tokens(const replay_config *cfg, int **prompt_out, int **expected_out) {
    FILE *file = fopen(cfg->token_file, "r");
    if (!file) {
        perror(cfg->token_file);
        return 0;
    }
    int *prompt = malloc((size_t)cfg->prompt_tokens * sizeof(*prompt));
    int *expected = malloc((size_t)cfg->steps * sizeof(*expected));
    if (!prompt || !expected) {
        fprintf(stderr, "token allocation failed\n");
        free(prompt);
        free(expected);
        fclose(file);
        return 0;
    }
    for (int i = 0; i < cfg->prompt_tokens + cfg->steps; i++) {
        int *token = i < cfg->prompt_tokens ? &prompt[i] : &expected[i - cfg->prompt_tokens];
        if (fscanf(file, "%d", token) != 1 || *token < 0) {
            fprintf(stderr, "invalid or missing token at position %d\n", i);
            free(prompt);
            free(expected);
            fclose(file);
            return 0;
        }
    }
    fclose(file);
    *prompt_out = prompt;
    *expected_out = expected;
    return 1;
}

static int sync_prompt(ds4_session *session, const ds4_tokens *tokens, const char *label) {
    char error[160] = {0};
    int rc = ds4_session_sync(session, tokens, error, sizeof(error));
    printf("{\"kind\":\"sync\",\"label\":\"%s\",\"tokens\":%d,\"position\":%d,\"rc\":%d}\n",
           label, tokens->len, ds4_session_pos(session), rc);
    if (rc != 0) fprintf(stderr, "%s sync failed: %s\n", label, error);
    return rc == 0;
}

/* Manifest mode stays separate to preserve the retained single-case evidence format. */

#define MANIFEST_FIELD_COUNT 8

static const char *manifest_header =
    "task_id\tattempt\tround_index\ttoken_file\tprompt_tokens\tsplit_at\tsteps\tthink";

typedef struct {
    char *task_id;
    int attempt;
    int round_index;
    char *token_file;
    int prompt_tokens;
    int split_at;
    int steps;
    int think;
} driver_case;

typedef struct {
    int done;
    int total;
} progress_chunk;

typedef struct {
    progress_chunk *items;
    size_t count;
    size_t capacity;
    int cached;
    int prefilled;
    int failed;
} progress_buffer;

typedef struct {
    int index;
    int position;
    int selected;
    int expected;
    double expected_logit;
    ds4_token_score top[TOP_COUNT];
} step_capture;

static double monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        return -1.0;
    }
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}

static void json_string(const char *value) {
    putchar('"');
    for (const unsigned char *cursor = (const unsigned char *)value; *cursor; cursor++) {
        switch (*cursor) {
            case '"': printf("\\\""); break;
            case '\\': printf("\\\\"); break;
            case '\b': printf("\\b"); break;
            case '\f': printf("\\f"); break;
            case '\n': printf("\\n"); break;
            case '\r': printf("\\r"); break;
            case '\t': printf("\\t"); break;
            default:
                if (*cursor < 0x20) printf("\\u%04x", (unsigned int)*cursor);
                else putchar(*cursor);
        }
    }
    putchar('"');
}

static int loaded_library_sha256(char output[65]) {
    Dl_info info;
    if (!dladdr((void *)&ds4_engine_open, &info) || !info.dli_fname) return 0;
    return reference_sha256_file(info.dli_fname, output);
}

static const char *observed_model_path(const replay_config *cfg, char resolved[4096]) {
    const char *path = realpath(cfg->model, resolved);
    return path ? path : cfg->model;
}

/* Parse and validate the shared TSV/NDJSON contract. */

static int parse_manifest_case(char *line, driver_case *out) {
    if (strchr(line, '\r')) {
        fprintf(stderr, "manifest contains a carriage return\n");
        return 0;
    }
    char *fields[MANIFEST_FIELD_COUNT];
    char *cursor = line;
    for (int index = 0; index < MANIFEST_FIELD_COUNT; index++) {
        fields[index] = cursor;
        char *tab = strchr(cursor, '\t');
        if (index + 1 < MANIFEST_FIELD_COUNT) {
            if (!tab) {
                fprintf(stderr, "manifest row has too few columns\n");
                return 0;
            }
            *tab = '\0';
            cursor = tab + 1;
        } else if (tab) {
            fprintf(stderr, "manifest row has too many columns\n");
            return 0;
        }
    }
    *out = (driver_case){.task_id = fields[0], .token_file = fields[3]};
    if (!out->task_id[0] || !out->token_file[0] ||
        !parse_number(fields[1], "attempt", &out->attempt) ||
        !parse_number(fields[2], "round_index", &out->round_index) ||
        !parse_number(fields[4], "prompt_tokens", &out->prompt_tokens) ||
        !parse_number(fields[5], "split_at", &out->split_at) ||
        !parse_number(fields[6], "steps", &out->steps) ||
        !parse_number(fields[7], "think", &out->think) ||
        out->prompt_tokens <= 0 || out->split_at > out->prompt_tokens) {
        fprintf(stderr, "invalid manifest row\n");
        return 0;
    }
    return 1;
}

static int manifest_think_mode(const driver_case *item, ds4_think_mode *out) {
    /* The TSV stores a native enum value, not the effort accepted by --think. */
    ds4_think_mode mode = (ds4_think_mode)item->think;
    if (mode == DS4_THINK_NONE || ds4_think_mode_enabled(mode) || ds4_think_mode_level(mode) >= 0) {
        *out = mode;
        return 1;
    }
    fprintf(stderr, "invalid native think mode: %d\n", item->think);
    return 0;
}

static int load_case_tokens(const driver_case *item, int **prompt_out, int **expected_out) {
    FILE *file = fopen(item->token_file, "r");
    if (!file) {
        perror(item->token_file);
        return 0;
    }
    int *prompt = malloc((size_t)item->prompt_tokens * sizeof(*prompt));
    int *expected = item->steps ? malloc((size_t)item->steps * sizeof(*expected)) : NULL;
    if (!prompt || (item->steps && !expected)) {
        fprintf(stderr, "token allocation failed\n");
        free(prompt);
        free(expected);
        fclose(file);
        return 0;
    }
    for (int index = 0; index < item->prompt_tokens + item->steps; index++) {
        int *token = index < item->prompt_tokens ? &prompt[index] : &expected[index - item->prompt_tokens];
        if (fscanf(file, "%d", token) != 1 || *token < 0) {
            fprintf(stderr, "invalid or missing token at position %d\n", index);
            free(prompt);
            free(expected);
            fclose(file);
            return 0;
        }
    }
    char extra;
    if (fscanf(file, " %c", &extra) == 1) {
        fprintf(stderr, "token file contains extra data\n");
        free(prompt);
        free(expected);
        fclose(file);
        return 0;
    }
    fclose(file);
    *prompt_out = prompt;
    *expected_out = expected;
    return 1;
}

static void manifest_progress(void *ud, const char *event, int current, int total) {
    (void)total;
    progress_buffer *buffer = ud;
    if (buffer->failed || strcmp(event, "prefill_chunk") != 0) return;
    int done = current - buffer->cached;
    if (done < 0) done = 0;
    if (done > buffer->prefilled) done = buffer->prefilled;
    if (buffer->count == buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity * 2 : 16;
        progress_chunk *items = realloc(buffer->items, capacity * sizeof(*items));
        if (!items) {
            buffer->failed = 1;
            return;
        }
        buffer->items = items;
        buffer->capacity = capacity;
    }
    buffer->items[buffer->count++] = (progress_chunk){.done = done, .total = buffer->prefilled};
}

static void emit_round_start(const replay_config *cfg, const driver_case *item, const char *library_sha256) {
    char resolved_model[4096];
    printf("{\"kind\":\"round_start\",\"source_trace\":");
    json_string(cfg->source_trace);
    printf(",\"task_id\":");
    json_string(item->task_id);
    printf(",\"attempt\":%d,\"round\":%d,\"token_file\":", item->attempt, item->round_index);
    json_string(item->token_file);
    printf(",\"prompt_tokens\":%d,\"split_at\":%d,\"steps\":%d,\"think\":%d,"
           "\"observed_libds4_sha256\":",
           item->prompt_tokens, item->split_at, item->steps, item->think);
    json_string(library_sha256);
    printf(",\"observed_model_path\":");
    json_string(observed_model_path(cfg, resolved_model));
    printf(",\"observed_backend\":");
    json_string(cfg->backend_name);
    printf(",\"observed_ctx\":%d,\"observed_seed\":1,\"observed_temperature\":0,"
           "\"observed_think\":%d}\n",
           cfg->ctx, item->think);
}

static int sync_case(
        ds4_session *session,
        const ds4_tokens *tokens,
        const char *label,
        progress_buffer *progress,
        int *prefill_tokens,
        double *prefill_seconds) {
    int common = ds4_session_common_prefix(session, tokens);
    int position = ds4_session_pos(session);
    if (common < 0) return 0;
    int cached = common == position ? common : 0;
    int prefilled = tokens->len - cached;
    size_t progress_start = progress->count;
    progress->cached = cached;
    progress->prefilled = prefilled;
    double start = monotonic_seconds();
    if (start < 0) return 0;
    char error[160] = {0};
    int rc = ds4_session_sync(session, tokens, error, sizeof(error));
    double end = monotonic_seconds();
    if (end < 0) return 0;
    double elapsed = end - start;
    printf("{\"kind\":\"sync\",\"label\":\"%s\",\"input_tokens\":%d,"
           "\"cached_tokens\":%d,\"prefill_tokens\":%d,\"position\":%d,"
           "\"elapsed_seconds\":%.9g}\n",
           label, tokens->len, cached, prefilled, ds4_session_pos(session), elapsed);
    for (size_t index = progress_start; index < progress->count; index++) {
        printf("{\"kind\":\"prefill\",\"done\":%d,\"total\":%d}\n",
               progress->items[index].done, progress->items[index].total);
    }
    if (progress->failed) {
        fprintf(stderr, "progress allocation failed\n");
        return 0;
    }
    if (rc != 0) {
        fprintf(stderr, "%s sync failed: %s\n", label, error);
        return 0;
    }
    *prefill_tokens += prefilled;
    *prefill_seconds += elapsed;
    return 1;
}

static void emit_step(const step_capture *step) {
    printf("{\"kind\":\"step\",\"index\":%d,\"position\":%d,"
           "\"selected_token\":%d,\"expected\":%d,\"expected_logit\":%.9g,\"top\":[",
           step->index, step->position, step->selected, step->expected, step->expected_logit);
    for (int index = 0; index < TOP_COUNT; index++) {
        printf("%s{\"token\":%d,\"logit\":%.9g}",
               index ? "," : "", step->top[index].id, step->top[index].logit);
    }
    printf("]}\n");
}

/* Execute one fresh native session per manifest case. */

static int run_manifest_case(
        ds4_engine *engine,
        const replay_config *cfg,
        const driver_case *item,
        const char *library_sha256) {
    int *prompt = NULL;
    int *expected = NULL;
    step_capture *steps = NULL;
    ds4_session *session = NULL;
    progress_buffer progress = {0};
    int ok = 0;
    if (!load_case_tokens(item, &prompt, &expected)) goto cleanup;
    ds4_think_mode think_mode;
    if (!manifest_think_mode(item, &think_mode)) goto cleanup;
    if (ds4_session_create(&session, engine, cfg->ctx) != 0) {
        fprintf(stderr, "failed to create native session\n");
        goto cleanup;
    }
    ds4_session_set_progress(session, manifest_progress, &progress);
    ds4_tokens full = {.v = prompt, .len = item->prompt_tokens, .cap = item->prompt_tokens};
    int prefill_tokens = 0;
    double prefill_seconds = 0.0;
    emit_round_start(cfg, item, library_sha256);
    if (item->split_at > 0) {
        ds4_tokens prefix = {.v = prompt, .len = item->split_at, .cap = item->split_at};
        if (!sync_case(session, &prefix, "prefix", &progress, &prefill_tokens, &prefill_seconds)) goto cleanup;
    }
    if (!sync_case(session, &full, "full", &progress, &prefill_tokens, &prefill_seconds)) goto cleanup;

    steps = calloc((size_t)(item->steps ? item->steps : 1), sizeof(*steps));
    if (!steps) {
        fprintf(stderr, "step allocation failed\n");
        goto cleanup;
    }
    uint64_t rng = 1;
    int generated = 0;
    const char *terminal_reason = "max_tokens";
    double decode_start = monotonic_seconds();
    if (decode_start < 0) goto cleanup;
    for (int index = 0; index < item->steps; index++) {
        step_capture *step = &steps[index];
        step->index = index;
        step->position = ds4_session_pos(session);
        step->expected = expected[index];
        int top_count = ds4_session_top_logprobs(session, step->top, TOP_COUNT);
        ds4_token_score expected_score;
        if (top_count != TOP_COUNT ||
            !ds4_session_token_logprob(session, expected[index], &expected_score)) {
            fprintf(stderr, "failed to capture top-12 logits at step %d\n", index);
            goto cleanup;
        }
        step->expected_logit = expected_score.logit;
        step->selected = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.05f, &rng);
        generated++;
        if (ds4_token_is_stop_for_think_mode(engine, step->selected, think_mode)) {
            terminal_reason = "stop_token";
            break;
        }
        char error[160] = {0};
        if (ds4_session_eval(session, step->selected, error, sizeof(error)) != 0) {
            fprintf(stderr, "decode failed at step %d: %s\n", index, error);
            goto cleanup;
        }
    }
    double decode_end = monotonic_seconds();
    if (decode_end < 0) goto cleanup;
    for (int index = 0; index < generated; index++) emit_step(&steps[index]);
    printf("{\"kind\":\"round_end\",\"prefill_tokens\":%d,\"decode_steps\":%d,"
           "\"prefill_seconds\":%.9g,\"decode_seconds\":%.9g,\"terminal_reason\":",
           prefill_tokens, generated, prefill_seconds, decode_end - decode_start);
    json_string(terminal_reason);
    printf("}\n");
    ok = 1;

cleanup:
    if (session) ds4_session_free(session);
    free(steps);
    free(progress.items);
    free(prompt);
    free(expected);
    return ok;
}

static int run_cases(const replay_config *cfg, const char *library_sha256) {
    FILE *file = fopen(cfg->cases_file, "r");
    if (!file) {
        perror(cfg->cases_file);
        return 1;
    }
    char *line = NULL;
    size_t line_capacity = 0;
    ds4_engine *engine = NULL;
    int exit_code = 1;
    if (getline(&line, &line_capacity, file) < 0) {
        fprintf(stderr, "empty case manifest\n");
        goto cleanup;
    }
    size_t length = strlen(line);
    if (length && line[length - 1] == '\n') line[length - 1] = '\0';
    if (strcmp(line, manifest_header) != 0) {
        fprintf(stderr, "invalid case manifest header\n");
        goto cleanup;
    }
    ds4_engine_options options = {
        .model_path = cfg->model,
        .backend = cfg->backend,
        .context_size = cfg->ctx,
        .placement_ctx_hint = cfg->ctx,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
    };
    if (ds4_engine_open(&engine, &options) != 0) {
        fprintf(stderr, "failed to open native engine\n");
        goto cleanup;
    }
    int case_count = 0;
    while (getline(&line, &line_capacity, file) >= 0) {
        length = strlen(line);
        if (length && line[length - 1] == '\n') line[length - 1] = '\0';
        driver_case item;
        if (!parse_manifest_case(line, &item)) goto cleanup;
        if (!run_manifest_case(engine, cfg, &item, library_sha256)) goto cleanup;
        case_count++;
    }
    if (!case_count) {
        fprintf(stderr, "case manifest contains no cases\n");
        goto cleanup;
    }
    exit_code = 0;

cleanup:
    if (engine) ds4_engine_close(engine);
    free(line);
    fclose(file);
    return exit_code;
}

/* Retained single-case investigation path. */

int main(int argc, char **argv) {
    if (setenv("DS4_CUDA_Q8_F16_CACHE_MB", Q8_F16_CACHE_MB, 1) != 0) {
        perror("failed to pin q8 fp16 cache budget");
        return 1;
    }
    replay_config cfg;
    if (!parse_options(argc, argv, &cfg)) {
        usage(stderr, argv[0]);
        return 2;
    }
    char library_sha256[65];
    if (!loaded_library_sha256(library_sha256)) {
        fprintf(stderr, "failed to identify or hash the loaded libds4\n");
        return 1;
    }
    if (cfg.cases_file) return run_cases(&cfg, library_sha256);
    ds4_think_mode think_mode = DS4_THINK_HIGH;
    if (strcmp(cfg.think_name, "high") != 0 &&
        !ds4_think_mode_parse_level(cfg.think_name, &think_mode)) {
        fprintf(stderr, "invalid think level: %s\n", cfg.think_name);
        return 2;
    }

    int *prompt = NULL;
    int *expected = NULL;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    int exit_code = 1;
    if (!load_tokens(&cfg, &prompt, &expected)) goto cleanup;

    ds4_engine_options options = {
        .model_path = cfg.model,
        .backend = cfg.backend,
        .context_size = cfg.ctx,
        .placement_ctx_hint = cfg.ctx,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
    };
    if (ds4_engine_open(&engine, &options) != 0) {
        fprintf(stderr, "failed to open native engine\n");
        goto cleanup;
    }
    if (ds4_session_create(&session, engine, cfg.ctx) != 0) {
        fprintf(stderr, "failed to create native session\n");
        goto cleanup;
    }

    ds4_tokens full = {.v = prompt, .len = cfg.prompt_tokens, .cap = cfg.prompt_tokens};
    if (cfg.split_at > 0) {
        ds4_tokens prefix = {.v = prompt, .len = cfg.split_at, .cap = cfg.split_at};
        if (!sync_prompt(session, &prefix, "prefix")) goto cleanup;
    }
    if (!sync_prompt(session, &full, "full")) goto cleanup;

    char resolved_model[4096];
    printf("{\"kind\":\"config\",\"model\":\"%s\",\"backend\":\"%s\",\"ctx\":%d,"
           "\"prompt_tokens\":%d,\"split_at\":%d,\"steps\":%d,\"temperature\":0,"
           "\"top_k\":0,\"top_p\":1,\"min_p\":0.05,\"seed\":1,"
           "\"observed_libds4_sha256\":\"%s\",\"observed_model_path\":",
           cfg.model, cfg.backend_name, cfg.ctx, cfg.prompt_tokens, cfg.split_at, cfg.steps,
           library_sha256);
    json_string(observed_model_path(&cfg, resolved_model));
    printf(",\"observed_backend\":");
    json_string(cfg.backend_name);
    printf(",\"observed_ctx\":%d,\"observed_seed\":1,\"observed_temperature\":0,"
           "\"observed_think\":%d}\n",
           cfg.ctx, (int)think_mode);

    uint64_t rng = 1;
    int first_divergence = -1;
    int generated = 0;
    for (int index = 0; index < cfg.steps; index++) {
        ds4_token_score top[TOP_COUNT];
        ds4_token_score expected_score;
        int top_count = ds4_session_top_logprobs(session, top, TOP_COUNT);
        if (top_count < 0 || !ds4_session_token_logprob(session, expected[index], &expected_score)) {
            fprintf(stderr, "failed to capture logits at step %d\n", index);
            goto cleanup;
        }
        uint64_t rng_before = rng;
        int position = ds4_session_pos(session);
        int token=ds4_session_sample(session, 0.0f, 0, 1.0f, 0.05f, &rng);
        printf("{\"kind\":\"step\",\"index\":%d,\"position\":%d,\"rng_before\":%llu,"
               "\"rng_after\":%llu,\"selected\":%d,\"expected\":%d,\"expected_logit\":%.9g,"
               "\"top\":[",
               index, position, (unsigned long long)rng_before, (unsigned long long)rng,
               token, expected[index], expected_score.logit);
        for (int i = 0; i < top_count && top[i].id >= 0; i++) {
            printf("%s{\"token\":%d,\"logit\":%.9g,\"logprob\":%.9g}",
                   i ? "," : "", top[i].id, top[i].logit, top[i].logprob);
        }
        printf("]}\n");
        generated++;
        if (first_divergence < 0 && token != expected[index]) first_divergence = index;
        if (ds4_token_is_stop_for_think_mode(engine, token, think_mode)) break;
        char error[160] = {0};
        if (ds4_session_eval(session, token, error, sizeof(error)) != 0) {
            fprintf(stderr, "decode failed at step %d: %s\n", index, error);
            goto cleanup;
        }
    }
    printf("{\"kind\":\"summary\",\"generated\":%d,\"first_divergence\":%d}\n",
           generated, first_divergence);
    exit_code = 0;

cleanup:
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    free(prompt);
    free(expected);
    return exit_code;
}
