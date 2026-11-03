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
 *   cc -O2 -I/workspace/ds4 bench/c_replay_diagnostic.c \
 *     -L/workspace/native -Wl,-rpath,/workspace/native -lds4 \
 *     -o /tmp/c-replay-diagnostic
 *
 * Then run with the case's prompt length and original cached-prefix length.
 * Native logs stay on stderr; stdout is NDJSON diagnostic evidence.
 */

#include "ds4.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOP_COUNT 12

typedef struct {
    const char *model;
    const char *token_file;
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
            "  --backend cuda|cpu   native backend (default: cuda)\n"
            "  --ctx N              context size (default: 100000)\n"
            "  --split-at N         prefill N tokens before the full prompt\n"
            "  --steps N            tokens to capture (default: 16)\n"
            "  --think LEVEL        native think level (default: high)\n",
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
    if (!cfg->model || !cfg->token_file || cfg->prompt_tokens <= 0 || cfg->ctx <= 0 ||
        cfg->steps <= 0 || cfg->split_at > cfg->prompt_tokens) {
        return 0;
    }
    return 1;
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

int main(int argc, char **argv) {
    replay_config cfg;
    if (!parse_options(argc, argv, &cfg)) {
        usage(stderr, argv[0]);
        return 2;
    }
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

    printf("{\"kind\":\"config\",\"model\":\"%s\",\"backend\":\"%s\",\"ctx\":%d,"
           "\"prompt_tokens\":%d,\"split_at\":%d,\"steps\":%d,\"temperature\":0,"
           "\"top_k\":0,\"top_p\":1,\"min_p\":0.05,\"seed\":1}\n",
           cfg.model, cfg.backend_name, cfg.ctx, cfg.prompt_tokens, cfg.split_at, cfg.steps);

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
