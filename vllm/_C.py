from argparse import Namespace

vllm_ops_list = [
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gelu_fast",
    "gelu_new",
    "paged_attention_v1",
    "paged_attention_v2",
    "rotary_embedding",
    "batched_rotary_embedding",
    "rms_norm",
    "fused_add_rms_norm",
    "awq_dequantize",
    "awq_gemm",
    "gptq_gemm",
    "gptq_shuffle",
    "squeezellm_gemm",
    "marlin_gemm",
    "scaled_fp8_quant",
    "moe_align_block_size",
]

vllm_cache_ops_list = [
    "reshape_and_cache",
    "copy_blocks",
    "swap_blocks",
    "convert_fp8",
]

def get_not_implemented_error_func(ops_name):
    def inner_func(*args, **kwargs):
        # import pdb; pdb.set_trace()
        raise NotImplementedError(f"Operation {ops_name} is not implemented, args: {args}, kwargs: {kwargs}")
    return inner_func

ops = Namespace(**{
   ops_name: get_not_implemented_error_func(ops_name) for ops_name in vllm_ops_list
})
cache_ops = Namespace(**{
   ops_name: get_not_implemented_error_func(ops_name) for ops_name in vllm_cache_ops_list
})
