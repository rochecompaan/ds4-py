/* Small ABI boundary double: never loads a model or runs inference. */
#include "ds4.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

struct ds4_session {
    ds4_tokens tokens;
    ds4_session_progress_fn progress, display;
    ds4_session_cancel_fn cancel;
    float logits[256];
    int logits_set;
    pthread_t owner;
};
static int frees, clears, snapshots, wrong_thread_frees, fail_snapshot;
static int copy_logits_result = 256;
void stub_copy_logits_result(int result) { copy_logits_result = result; }
int stub_wrong_thread_frees(void) { return wrong_thread_frees; }
void stub_fail_snapshot(int fail) { fail_snapshot = fail; }
int stub_frees(void) { return frees; }
int stub_clears(void) { return clears; }
int stub_snapshots(void) { return snapshots; }
int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx) {
    if (e != (ds4_engine *)(uintptr_t)0x1234 || ctx != 64) return 1;
    *out = calloc(1, sizeof(**out));
    (*out)->owner = pthread_self();
    return 0;
}
void ds4_session_free(ds4_session *s) {
    if (!pthread_equal(s->owner, pthread_self())) wrong_thread_frees++;
    free(s->tokens.v); free(s); frees++;
}
void ds4_engine_close(ds4_engine *e) { (void)e; }
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    (void)ud; s->progress = fn; if (!fn) clears++;
}
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    (void)ud; s->display = fn; if (!fn) clears++;
}
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    (void)ud; s->cancel = fn; if (!fn) clears++;
}
int ds4_session_sync(ds4_session *s, const ds4_tokens *t, char *err, size_t n) {
    if (!t->len) { snprintf(err, n, "empty prompt"); return 1; }
    if (s->display) s->display(NULL, "prefill_display", 0, t->len);
    if (s->progress) s->progress(NULL, "prefill_chunk", t->len, t->len);
    if (s->cancel && s->cancel(NULL)) return 2;
    free(s->tokens.v);
    s->tokens.v = malloc(t->len * sizeof(int));
    memcpy(s->tokens.v, t->v, t->len * sizeof(int)); s->tokens.len = t->len;
    return 0;
}
const ds4_tokens *ds4_session_tokens(ds4_session *s) { return &s->tokens; }
int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    if (cap != 256) return -1;
    if (!s->logits_set) for (int i = 0; i < 256; i++) s->logits[i] = (float)i;
    memcpy(out, s->logits, sizeof(s->logits));
    return copy_logits_result;
}
int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    if (n != 256) return 1;
    memcpy(s->logits, logits, sizeof(s->logits)); s->logits_set = 1;
    return 0;
}
int ds4_session_eval(ds4_session *s, int token, char *err, size_t n) {
    (void)s; (void)token; snprintf(err, n, "eval rejected"); return 1;
}
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t n) {
    (void)s; snap->ptr = malloc(4); snap->cap = 4;
    if (fail_snapshot) { snprintf(err, n, "snapshot write rejected"); return 1; }
    snap->len = 4; return 0;
}
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t n) {
    (void)s; (void)snap; (void)err; (void)n; return 0;
}
void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (snap->ptr) snapshots++;
    free(snap->ptr); memset(snap, 0, sizeof(*snap));
}
int ds4_session_sample(ds4_session *s, float t, int k, float p, float m, uint64_t *rng) {
    (void)s; if (t != 0.5f || k != 7 || p != 0.75f || m != 0.25f) return -1;
    *rng += 8; return 42;
}
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e; (void)text; out->v = malloc(2*sizeof(int)); out->v[0]=7; out->v[1]=9; out->len=out->cap=2;
}
void ds4_tokens_free(ds4_tokens *t) { free(t->v); memset(t, 0, sizeof(*t)); }
char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e; (void)token; char *p=malloc(3); memcpy(p,"a\0b",3); *len=3; return p;
}
static char opened_lock[1024];
const char *stub_opened_lock(void) { return opened_lock; }
int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    snprintf(opened_lock, sizeof(opened_lock), "%s", getenv("DS4_LOCK_FILE"));
    if (opt->context_size == 13) return 1;
    *out = (ds4_engine *)(uintptr_t)0x1234;
    return 0;
}
static int info_reads;
int stub_info_reads(void) { return info_reads; }
const char *ds4_engine_model_name(ds4_engine *e) { (void)e; info_reads++; return "stub-model"; }
int ds4_engine_model_id(ds4_engine *e) { (void)e; return 17; }
bool ds4_engine_is_deepseek41(ds4_engine *e) { (void)e; return false; }
bool ds4_engine_is_glm_dsa(ds4_engine *e) { (void)e; return true; }
bool ds4_engine_is_qwen4(ds4_engine *e) { (void)e; return false; }
bool ds4_engine_has_vision(ds4_engine *e) { (void)e; return false; }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; return 4; }
int ds4_engine_mtp_draft_tokens(ds4_engine *e) { (void)e; return 1; }
int ds4_engine_vocab_size(ds4_engine *e) { (void)e; return 256; }
static unsigned char staged_payload[4096];
static size_t staged_size;
static int payload_frees, payload_writes, payload_loads;
static int fail_stage, fail_write, fail_load;
void stub_fail_payload(int stage, int write, int load) {
    fail_stage = stage; fail_write = write; fail_load = load;
}
int stub_payload_frees(void) { return payload_frees; }
int stub_payload_writes(void) { return payload_writes; }
int stub_payload_loads(void) { return payload_loads; }
uint64_t ds4_session_payload_bytes(ds4_session *s) {
    return 4u + 4u * (uint64_t)s->tokens.len;
}
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out, char *err, size_t n) {
    size_t size = 4u + 4u * (size_t)s->tokens.len;
    if (fail_stage) { snprintf(err, n, "stage rejected: checkpoint is invalid"); return 1; }
    if (size > sizeof(staged_payload)) { snprintf(err, n, "stage rejected: payload too large"); return 1; }
    memcpy(staged_payload, &s->tokens.len, 4);
    memcpy(staged_payload + 4, s->tokens.v, size - 4);
    staged_size = size;
    out->path = strdup("stub-payload.bin");
    out->bytes = size;
    return 0;
}
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t n) {
    if (fail_write) { snprintf(err, n, "write rejected: disk full"); return 1; }
    if (!payload->path || payload->bytes != staged_size) {
        snprintf(err, n, "write rejected: unknown staged payload"); return 1;
    }
    if (fwrite(staged_payload, 1, staged_size, fp) != staged_size) {
        snprintf(err, n, "write rejected: short write"); return 1;
    }
    payload_writes++;
    return 0;
}
void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    payload_frees++;
    free(payload->path); memset(payload, 0, sizeof(*payload));
}
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                             char *err, size_t n) {
    unsigned char buffer[4096];
    int count;
    if (fail_load) { snprintf(err, n, "load rejected: corrupt payload"); return 1; }
    if (payload_bytes < 4 || payload_bytes % 4 || payload_bytes > sizeof(buffer)) {
        snprintf(err, n, "load rejected: bad payload size"); return 1;
    }
    if (fread(buffer, 1, (size_t)payload_bytes, fp) != payload_bytes) {
        snprintf(err, n, "load rejected: truncated payload"); return 1;
    }
    memcpy(&count, buffer, 4);
    if (count < 0 || 4u + 4u * (size_t)count != payload_bytes) {
        snprintf(err, n, "load rejected: token count mismatch"); return 1;
    }
    free(s->tokens.v);
    s->tokens.v = malloc(4u * (size_t)count);
    memcpy(s->tokens.v, buffer + 4, 4u * (size_t)count);
    s->tokens.len = s->tokens.cap = count;
    payload_loads++;
    return 0;
}
