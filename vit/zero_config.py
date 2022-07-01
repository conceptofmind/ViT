from colossalai.amp import AMP_TYPE
from colossalai.zero.shard_utils import TensorShardStrategy

# Colossal AI Global Config

zero = dict(
    model_config = dict(
        shard_strategy = TensorShardStrategy(),
        tensor_placement_policy = 'auto',
        reuse_fp16_shard = False
    ),
    optimizer_config=dict(
        gpu_margin_mem_ratio = 0.8,
        initial_scale=2**5,
    )
)

gradient_accumulation = 4
clip_grad_norm = 1.0