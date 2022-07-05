from colossalai.amp import AMP_TYPE

fp16 = dict(
    mode = AMP_TYPE.TORCH,
    init_scale = 2.**16,
    growth_factor = 2.0,
    backoff_factor = 0.5,
    growth_interval = 2000,
    enabled = True
)

gradient_accumulation = 16
clip_grad_norm = 1.0