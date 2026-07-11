#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100BF16PagedMQALogitsRuntime final: public LaunchRuntime<SM100BF16PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        bool is_varlen;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;

        int* context_lens;
        void* logits;
        int* block_table;
        int* indices;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_bf16_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_bf16_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", arch, arch,
    args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits,
            args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_bf16_paged_mqa_logits(const torch::Tensor& q,
                                      const torch::Tensor& kv_cache,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& context_lens,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& indices,
                                      const torch::Tensor& schedule_meta,
                                      const at::ScalarType& logits_dtype,
                                      const int& batch_size, const int& next_n,
                                      const int& num_heads, const int& head_dim,
                                      const int& num_kv_blocks, const int& block_kv,
                                      const bool& is_context_lens_2d,
                                      const bool& is_varlen,
                                      const int& logits_stride,
                                      const int& block_table_stride,
                                      const int& num_sms,
                                      const int& split_kv) {
    const int num_specialized_threads = 128;
    const int mma_m = (device_runtime->get_arch_major() == 10 ? 128 : 64);
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 128;
    const int num_q_stages = 2, num_kv_stages = (device_runtime->get_arch_major() == 10 ? 3 : 2);
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);

    // Construct TMAs
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               head_dim);
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                head_dim);

    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);

    // Calculate shared memory size
    int smem_size = 0;
    if (device_runtime->get_arch_major() == 9) {
        const int swizzle_alignment = head_dim * 8;

        const int smem_q_size_per_stage = next_n * num_heads * head_dim * static_cast<int>(q.element_size());
        const int aligned_smem_weight_size_per_stage = align(next_n * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
        const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) + align(num_q_stages * 8 * 2, swizzle_alignment);

        const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int smem_kv_pipe_size = num_kv_stages * smem_kv_size_per_stage + align(num_kv_stages * 8 * 2, swizzle_alignment);

        // Allocate some shared memory for UMMA barriers and tensor memory pointer, although it is not used in SM90
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
        DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    } else {
        const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim * static_cast<int>(q.element_size());
        const int smem_kv_size_per_stage = split_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int smem_weight_size_per_stage = next_n_atom * num_heads * static_cast<int>(weights.element_size());

        const int smem_barriers = (num_q_stages + num_kv_stages) * 2 * 8;
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = num_q_stages * (smem_q_size_per_stage + smem_weight_size_per_stage) +
                    num_kv_stages * smem_kv_size_per_stage +
                    smem_barriers + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);
    }

    // Launch
    const SM100BF16PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .is_varlen = is_varlen,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .indices = is_varlen ? indices.data_ptr<int>() : nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100BF16PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_paged_mqa_logits", code);
    SM100BF16PagedMQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
