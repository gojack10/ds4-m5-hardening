struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    uint32_t causal_start;
    uint32_t causal_ratio;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
    int32_t  total;
    int32_t  keep_k;
    uint32_t causal_start;
    uint32_t causal_ratio;
    uint32_t block_width;
    uint32_t block_top_k;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_f32_i32_desc_causal")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_f32_i32_desc_causal_shuffle")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order, bool causal = false, bool prefix = false>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    // Causal selection retains upstream offsets; ordinary runs pack at keep_k.
    const int start_read = im * (2 * args.len);
    const int new_len = causal ? 2 * args.len : MIN(2 * args.len, args.keep_k);
    const int start_write = im * new_len;
    const uint width = causal ? min(uint(args.ne00),
        (args.causal_start + uint(i01) + 1u) / args.causal_ratio) : 0u;
    const int work_width = causal ? int(((width - 1u) / args.block_width) * args.block_top_k +
        min((width - 1u) % args.block_width + 1u, args.block_top_k)) : args.total;
    const int read_limit = prefix ? min(args.len, 512) : args.len;
    const int len0 = MIN(read_limit, MAX(0, work_width - start_read));
    const int len1 = MIN(read_limit, MAX(0, work_width - (start_read + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start_read
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start_write
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int out = causal ? MIN(total, prefix ? min(args.top_k, 512) : args.top_k)
                           : MIN(total, new_len);
    const int chunk = ((causal ? total : out) + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(k0 + chunk, out);

    if (k0 >= out) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal_prefix")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

// ---- Radix-select top-k -------------------------------------------------
// The bitonic sort above keeps every block's top block_top_k rows and merges
// them in log2(blocks) passes: for a 435k-row indexer score row that is a
// near-complete sort, nine dependent passes, ~1 ms per token per index layer.
// A selection does not need the order: four 8-bit histogram passes find the
// k-th largest key exactly, one pass counts the ties at it per chunk, one
// compaction pass gathers the rows above it and the first ties by index, and
// a final threadgroup sorts the k winners so the output is in descending
// score order like the sort's.
//
// Per-token state (uint words): [0] prefix (key bits settled so far, left
// aligned), [1] above (rows with key > prefix so far), [2] need (rows still
// to take from the current prefix), [3] sel_count, [5] level, [16..272)
// histogram of the current byte, then the per-chunk tie counts.
#define DS4_TOPK_STATE_WORDS 272u
#define DS4_TOPK_HIST_OFF 16u

struct ds4_metal_args_topk_select {
    uint n;             // row width
    uint n_tokens;
    uint top_k;
    uint causal_start;  // width per token = min(n, (causal_start + t + 1) / causal_ratio) when causal_ratio != 0
    uint causal_ratio;
    uint chunk;         // rows per threadgroup in the histogram and compaction passes
    uint level;         // histogram byte: 0 (top) .. 3
    uint64_t row_stride;   // score row stride in bytes
    uint out_stride;       // selected row stride in elements
};

static inline uint ds4_topk_width(constant ds4_metal_args_topk_select &args, uint t) {
    if (args.causal_ratio == 0u) return args.n;
    return min(args.n, (args.causal_start + t + 1u) / args.causal_ratio);
}

// Larger score -> larger key; -inf sorts last, and equal scores compare equal.
static inline uint ds4_topk_key(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

kernel void kernel_topk_select_init(
        constant ds4_metal_args_topk_select &args,
        device uint *state,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const uint t = tgpig;
    if (t >= args.n_tokens) return;
    device uint *st = state + t * DS4_TOPK_STATE_WORDS;
    for (uint i = tid; i < DS4_TOPK_STATE_WORDS; i += ntg) st[i] = 0u;
    if (tid == 0) st[2] = min(args.top_k, ds4_topk_width(args, t));
}

kernel void kernel_topk_select_hist(
        constant ds4_metal_args_topk_select &args,
        device const char *scores,
        device uint *state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg3 [[threads_per_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    const uint tid = tpitg.x, ntg = ntg3.x;
    const uint t = tgpig.y;
    const uint width = ds4_topk_width(args, t);
    const uint begin = tgpig.x * args.chunk;
    if (begin >= width) return;
    const uint end = min(begin + args.chunk, width);
    device uint *st = state + t * DS4_TOPK_STATE_WORDS;
    for (uint i = tid; i < 256u; i += ntg) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint level = args.level;
    const uint shift = 24u - 8u * level;
    const uint prefix = st[0];
    const uint pmask = level == 0u ? 0u : (0xFFFFFFFFu << (32u - 8u * level));
    device const float *row = (device const float *)(scores + (uint64_t)t * args.row_stride);
    for (uint i = begin + tid; i < end; i += ntg) {
        const uint key = ds4_topk_key(row[i]);
        if ((key & pmask) == prefix)
            atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 256u; i += ntg) {
        const uint c = atomic_load_explicit(&hist[i], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit((device atomic_uint *)(st + DS4_TOPK_HIST_OFF + i), c, memory_order_relaxed);
    }
}

// One 256-thread threadgroup per token: walk the histogram from the top bin
// down to the bin holding the need-th remaining row, settle that byte of the
// prefix, and clear the histogram for the next level.
kernel void kernel_topk_select_scan(
        constant ds4_metal_args_topk_select &args,
        device uint *state,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sums[8];
    threadgroup uint pick[3];
    const uint t = tgpig;
    if (t >= args.n_tokens) return;
    device uint *st = state + t * DS4_TOPK_STATE_WORDS;
    const uint bin = 255u - tid;                       // descending bins
    const uint c = st[DS4_TOPK_HIST_OFF + bin];
    const uint incl_sg = simd_prefix_inclusive_sum(c);
    if (lane == 31) sums[sg] = incl_sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    for (uint i = 0; i < sg; i++) before += sums[i];
    const uint incl = before + incl_sg;
    const uint excl = incl - c;
    const uint need = st[2];
    if (c != 0u && excl < need && need <= incl) {
        pick[0] = bin;
        pick[1] = excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint level = args.level;
        const uint shift = 24u - 8u * level;
        st[0] |= pick[0] << shift;
        st[1] += pick[1];
        st[2] = need - pick[1];
        st[5] = level + 1u;
    }
    st[DS4_TOPK_HIST_OFF + bin] = 0u;
}

// Count the ties at the threshold per chunk, so the compaction can rank
// them by index across the row without a tie buffer or a cap: the chunk
// counts go where the (now cleared) histogram lived, one word per chunk.
kernel void kernel_topk_select_tiecount(
        constant ds4_metal_args_topk_select &args,
        device const char *scores,
        device uint *state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg3 [[threads_per_threadgroup]]) {
    threadgroup atomic_uint count;
    const uint tid = tpitg.x, ntg = ntg3.x;
    const uint t = tgpig.y;
    const uint width = ds4_topk_width(args, t);
    const uint begin = tgpig.x * args.chunk;
    if (begin >= width) return;
    const uint end = min(begin + args.chunk, width);
    device uint *st = state + t * DS4_TOPK_STATE_WORDS;
    if (tid == 0) atomic_store_explicit(&count, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint thr = st[0];
    device const float *row = (device const float *)(scores + (uint64_t)t * args.row_stride);
    uint mine = 0u;
    for (uint i = begin + tid; i < end; i += ntg) mine += ds4_topk_key(row[i]) == thr ? 1u : 0u;
    mine = simd_sum(mine);
    if ((tid & 31u) == 0u && mine) atomic_fetch_add_explicit(&count, mine, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) st[DS4_TOPK_HIST_OFF + tgpig.x] = atomic_load_explicit(&count, memory_order_relaxed);
}

// Rows above the threshold go to the output's first `above` slots (any
// order: the finish sorts), and the ties take the slots after them in index
// order -- the first `need` ties of the row, exactly the ones an index-
// stable sort would keep. A chunk ranks its ties with a running prefix over
// strips of ntg rows, on top of the preceding chunks' counts.
kernel void kernel_topk_select_compact(
        constant ds4_metal_args_topk_select &args,
        device const char *scores,
        device uint *state,
        device int *selected,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg3 [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sums[32];
    const uint tid = tpitg.x, ntg = ntg3.x;
    const uint t = tgpig.y;
    const uint width = ds4_topk_width(args, t);
    const uint begin = tgpig.x * args.chunk;
    if (begin >= width) return;
    const uint end = min(begin + args.chunk, width);
    device uint *st = state + t * DS4_TOPK_STATE_WORDS;
    const uint thr = st[0];
    const uint above = st[1];
    const uint need = st[2];
    uint base = 0u;
    for (uint c = 0; c < tgpig.x; c++) base += st[DS4_TOPK_HIST_OFF + c];
    device const float *row = (device const float *)(scores + (uint64_t)t * args.row_stride);
    device int *out = selected + t * args.out_stride;
    const uint nsg = (ntg + 31u) / 32u;
    uint running = base;
    for (uint s = begin; s < end; s += ntg) {
        const uint i = s + tid;
        const bool in_row = i < end;
        const uint key = in_row ? ds4_topk_key(row[i]) : 0u;
        const uint tie = (in_row && key == thr) ? 1u : 0u;
        if (in_row && key > thr) {
            const uint slot = atomic_fetch_add_explicit((device atomic_uint *)(st + 3), 1u, memory_order_relaxed);
            if (slot < above) out[slot] = (int)i;
        }
        const uint excl = simd_prefix_exclusive_sum(tie);
        const uint sg_total = simd_sum(tie);
        if (lane == 0) sums[sg] = sg_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = 0u, total = 0u;
        for (uint g = 0; g < nsg; g++) { if (g < sg) before += sums[g]; total += sums[g]; }
        if (tie) {
            const uint rank = running + before + excl;
            if (rank < need) out[above + rank] = (int)i;
        }
        running += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One threadgroup per token: sort the k winners by score descending (index
// ascending among equals) so the row reads like the merge sort's output.
// top_k <= 2 * threads.
kernel void kernel_topk_select_finish(
        constant ds4_metal_args_topk_select &args,
        device const char *scores,
        device uint *state,
        device int *selected,
        threadgroup int *sh_idx [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const uint t = tgpig;
    if (t >= args.n_tokens) return;
    device const float *row = (device const float *)(scores + (uint64_t)t * args.row_stride);
    device int *out = selected + t * args.out_stride;
    const uint width = ds4_topk_width(args, t);
    const uint k = min(args.top_k, width);
    threadgroup float *sh_val = (threadgroup float *)(sh_idx + 2u * ntg);
    uint cap;
    // Order the winners: score descending, index ascending among equals.
    cap = 1u;
    while (cap < k) cap <<= 1u;
    for (uint i = tid; i < cap; i += ntg) {
        const int idx = i < k ? out[i] : 0x7FFFFFFF;
        sh_idx[i] = idx;
        sh_val[i] = i < k ? row[idx] : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2u; size <= cap; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (uint i = tid; i < cap; i += ntg) {
                const uint j = i ^ stride;
                if (j > i) {
                    const bool up = (i & size) == 0u;      // ascending run: we want descending, so flip
                    const float va = sh_val[i], vb = sh_val[j];
                    const int ia = sh_idx[i], ib = sh_idx[j];
                    // "a before b" in the final order: higher score, or equal score and lower index.
                    const bool a_first = va > vb || (va == vb && ia < ib);
                    if (up ? !a_first && !(va == vb && ia == ib) : a_first) {
                        sh_val[i] = vb; sh_val[j] = va;
                        sh_idx[i] = ib; sh_idx[j] = ia;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < k; i += ntg) out[i] = sh_idx[i];
}
