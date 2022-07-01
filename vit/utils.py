import torch

import psutil

from colossalai.trainer.hooks import BaseHook
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.logging import get_dist_logger

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

class TotalBatchsizeHook(BaseHook):
    def __init__(self, priority: int = 2) -> None:
        super().__init__(priority)
        self.logger = get_dist_logger()

    def before_train(self, trainer):
        total_batch_size = gpc.config.BATCH_SIZE * \
            gpc.config.gradient_accumulation * gpc.get_world_size(ParallelMode.DATA)
        self.logger.info(f'Total batch size = {total_batch_size}', ranks=[0])