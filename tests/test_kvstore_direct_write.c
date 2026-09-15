/* Model-free stream regression. From the repo root on macOS:
 * cc -std=c99 -D_GNU_SOURCE -Wall -Wextra -ffunction-sections -fdata-sections \
 *   tests/test_kvstore_direct_write.c -Wl,-dead_strip -o /tmp/test-kv-direct && /tmp/test-kv-direct
 * Linux: replace -Wl,-dead_strip with -Wl,--gc-sections.
 */
#include "../ds4_kvstore.c"
#include <assert.h>

static int fail_payload;
static int sparse_payload;
static int saves;
static uint64_t predicted_bytes = 7;
static int token = 42;
static ds4_tokens live = {.v = &token, .len = 1, .cap = 1};
int ds4_engine_model_id(ds4_engine *e) { (void)e; return 1; }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; return 4; }
int ds4_session_ctx(ds4_session *s) { (void)s; return 2048; }
const ds4_tokens *ds4_session_tokens(ds4_session *s) { (void)s; return &live; }
uint64_t ds4_session_payload_bytes(ds4_session *s) { (void)s; return predicted_bytes; }
void ds4_tokens_push(ds4_tokens *t, int value) {
    t->v = realloc(t->v, (size_t)(t->len + 1) * sizeof(int));
    assert(t->v);
    t->v[t->len++] = value;
    t->cap = t->len;
}
void ds4_tokens_free(ds4_tokens *t) { free(t->v); memset(t, 0, sizeof(*t)); }
bool ds4_tokens_starts_with(const ds4_tokens *t, const ds4_tokens *p) {
    return t->len >= p->len && !memcmp(t->v, p->v, (size_t)p->len * sizeof(int));
}
char *ds4_token_text(ds4_engine *e, int t, size_t *n) {
    (void)e; (void)t; *n = 4; return strdup("TEXT");
}
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t len) {
    (void)s;
    saves++;
    if (sparse_payload && fseeko(fp, (off_t)UINT32_MAX + 17, SEEK_CUR)) return -1;
    if (fwrite("PAYLOAD", 1, 7, fp) != 7) return -1;
    if (fail_payload) {
        snprintf(err, len, "injected serializer failure");
        return -1;
    }
    return 0;
}

static void check_file(int sparse) {
    char path[] = "/tmp/ds4-kv-direct-test.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb"); /* No reads: wb supports backpatching. */
    assert(fp);
    uint8_t header[48], expected[48], text_size[4];
    ds4_kvstore_fill_header(header, 1, 4, 2, DS4_KVSTORE_EXT_TOOL_MAP,
                            123, 9, 2048, 111, 222, 0);
    ds4_kvstore_le_put32(text_size, 4);
    assert(fwrite(header, 1, 48, fp) == 48);
    assert(fwrite(text_size, 1, 4, fp) == 4);
    assert(fwrite("TEXT", 1, 4, fp) == 4);
    uint64_t bytes = 0;
    char err[128] = {0};
    sparse_payload = sparse;
    const uint64_t want = 7 + (sparse ? (uint64_t)UINT32_MAX + 17 : 0);
    assert(ds4_kvstore_write_payload(fp, NULL, &bytes, err, sizeof(err)));
    assert(bytes == want);
    assert((uint64_t)ftello(fp) == 56 + want);
    assert(fwrite("TRAILER", 1, 7, fp) == 7);
    assert(fclose(fp) == 0);
    fp = fopen(path, "rb");
    assert(fp);
    ds4_kvstore_fill_header(expected, 1, 4, 2, DS4_KVSTORE_EXT_TOOL_MAP,
                            123, 9, 2048, 111, 222, want);
    assert(fread(header, 1, 48, fp) == 48);
    assert(memcmp(header, expected, 48) == 0);
    char buf[14];
    assert(fseeko(fp, (off_t)(56 + want - 7), SEEK_SET) == 0);
    assert(fread(buf, 1, sizeof(buf), fp) == sizeof(buf));
    assert(memcmp(buf, "PAYLOADTRAILER", 14) == 0);
    assert(fgetc(fp) == EOF);
    assert(fclose(fp) == 0);
    assert(unlink(path) == 0);
}

#ifdef __APPLE__
struct fault_stream { fpos_t pos; int fault; };
static int fault_write(void *cookie, const char *buf, int size) {
    (void)buf;
    struct fault_stream *s = cookie;
    if (s->fault == 1 || (s->fault == 4 && s->pos >= 44 && s->pos < 48)) {
        errno = ENOSPC;
        return -1;
    }
    if (s->fault == 4 && s->pos == 40) {
        s->pos += size / 2;
        return size / 2;
    }
    s->pos += size;
    return size;
}
static fpos_t fault_seek(void *cookie, fpos_t offset, int whence) {
    struct fault_stream *s = cookie;
    if (whence == SEEK_SET && ((s->fault == 2 && offset == 40) ||
                               (s->fault == 3 && offset == 59))) {
        errno = EIO;
        return -1;
    }
    s->pos = whence == SEEK_CUR ? s->pos + offset : offset;
    return s->pos;
}
static void check_stream_faults(void) {
    for (int fault = 1; fault <= 4; fault++) {
        struct fault_stream s = {.pos = 52, .fault = fault};
        FILE *fp = funopen(&s, NULL, fault_write, fault_seek, NULL);
        assert(fp);
        uint64_t bytes = 0;
        char err[128] = {0};
        assert(!ds4_kvstore_write_payload(fp, NULL, &bytes, err, sizeof(err)));
        assert(bytes == 0);
        fclose(fp);
    }
}
#endif

static bool trailer_size(void *ud, const char *text, uint64_t *n) {
    (void)ud; (void)text; *n = 7; return true;
}
static bool trailer_write(void *ud, FILE *fp, const char *text, uint64_t *n) {
    (void)text;
    *n = ud ? 100 : 7; /* Exercise final, not just estimated, budget check. */
    for (uint64_t i = 0; i < *n; i++) if (fputc('T', fp) == EOF) return false;
    return true;
}
static void check_store(uint64_t prediction, uint64_t budget, int fail, int big_trailer) {
    char dir[] = "/tmp/ds4-kv-store-test.XXXXXX";
    assert(mkdtemp(dir));
    ds4_kvstore kc = {.enabled = true, .dir = dir, .budget_bytes = budget,
                       .opt = {.min_tokens = 1}};
    ds4_kvstore_trailer_hooks hooks = {.ext_flag = DS4_KVSTORE_EXT_TOOL_MAP,
        .serialized_size = trailer_size, .write = trailer_write,
        .ud = big_trailer ? &kc : NULL};
    predicted_bytes = prediction;
    fail_payload = fail;
    char err[128] = {0};
    int before = saves;
    bool ok = ds4_kvstore_store_live_prefix_text(&kc, NULL, NULL, &live, 1,
        "continued", "TEXT", 0, NULL, NULL, &hooks, err, sizeof(err));
    bool expected = !fail && (!budget || budget >= (big_trailer ? 165 : 71));
    assert(ok == expected);
    if (prediction && budget && prediction + 63 > budget) assert(saves == before);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex("TEXT", 4, sha);
    char *path = ds4_kvstore_path_for_sha(&kc, sha);
    if (ok) {
        ds4_kvstore_entry entry = {0};
        assert(ds4_kvstore_read_entry_file(path, sha, &entry));
        assert(entry.payload_bytes == 7 && entry.file_size == 70);
        ds4_kvstore_entry_free(&entry);
        assert(unlink(path) == 0);
    } else assert(access(path, F_OK) != 0);
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    assert(access(tmp, F_OK) != 0);
    free(path);
    ds4_kvstore_clear(&kc);
    assert(rmdir(dir) == 0);
    fail_payload = 0;
}

int main(void) {
    check_file(0);
    check_file(1); /* Sparse >4GiB offset; no large allocation or payload write. */
    sparse_payload = 0;
    FILE *fp = tmpfile();
    assert(fp);
    uint8_t header[52] = {0};
    assert(fwrite(header, 1, sizeof(header), fp) == sizeof(header));
    uint64_t bytes = 0;
    char err[128] = {0};
    fail_payload = 1;
    assert(!ds4_kvstore_write_payload(fp, NULL, &bytes, err, sizeof(err)));
    assert(bytes == 0 && strstr(err, "injected"));
    assert(fseeko(fp, 40, SEEK_SET) == 0);
    uint8_t field[8];
    assert(fread(field, 1, 8, fp) == 8 && kv_le_get64(field) == 0);
    assert(fclose(fp) == 0);
    fail_payload = 0;
    int pipes[2];
    assert(pipe(pipes) == 0);
    fp = fdopen(pipes[1], "wb");
    assert(fp);
    assert(!ds4_kvstore_write_payload(fp, NULL, &bytes, err, sizeof(err)));
    assert(fclose(fp) == 0);
    close(pipes[0]);
#ifdef __APPLE__
    check_stream_faults();
#endif
    check_store(7, 100, 0, 0);
    check_store(0, 100, 0, 0); /* Distributed/unknown prediction. */
    check_store(3, 100, 0, 0); /* Actual differs from prediction. */
    check_store(7, 100, 1, 0); /* Partial serializer failure: no publication. */
    check_store(1000, 50, 0, 0); /* Predicted rejection: no serialization. */
    check_store(0, 50, 0, 0); /* Measured rejection: temp removed. */
    check_store(7, 100, 0, 1); /* Larger actual trailer: no publication. */
    puts("direct checkpoint stream/store tests: ok");
    return 0;
}
