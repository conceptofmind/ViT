from colossalai.amp import AMP_TYPE

# from colossalai.zero.shard_utils import TensorShardStrategy

# Colossal AI Global Config

fp16 = dict(
    mode = AMP_TYPE.TORCH,
    init_scale = 2.**16,
    growth_factor = 2.0,
    backoff_factor = 0.5,
    growth_interval = 2000,
    enabled = True
)

# zero = dict(
#     model_config = dict(
#         shard_strategy = TensorShardStrategy(),
#         tensor_placement_policy = 'cpu',
#         reuse_fp16_shard = False
#     )
# )

gradient_accumulation = 16
clip_grad_norm = 1.0