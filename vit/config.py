from typing import Optional
from dataclasses import dataclass, field

@dataclass
class CFG:

    """
    Configuration for ZeRO Offloading.
    """

    use_zero: bool = field(
        default = False,
        metadata = {'help': 'whether to use ZeRO Offloading'}
    )

    use_fp16: bool = field(
        default = False,
        metadata = {'help': 'whether to use FP16'}
    )

    """
    Configuration for Vision Transformer Model.
    """

    image_size: int = field(
        default = 224,
        metadata = {"help": "Image size."}
    )

    patch_size: int = field(
        default = 32,
        metadata = {"help": "Patch size."}
    )

    num_classes: int = field(
        default = 10,
        metadata = {"help": "Number of classes."}
    )

    dim: int = field(
        default = 1024,
        metadata = {"help": "Dimension of the embedding."}
    )

    depth: int = field(
        default = 6,
        metadata = {"help": "Depth of the transformer."}
    )

    heads: int = field(
        default = 16,
        metadata = {"help": "Number of heads in the transformer."}
    )

    mlp_dim: int = field(
        default = 2048,
        metadata = {"help": "Dimension of the MLP Layer."}
    )

    dropout: float = field(
        default = 0.1,
        metadata = {"help": "Dropout probability."}
    )

    emb_dropout: float = field(
        default = 0.1,
        metadata = {"help": "Embedding Dropout probability."}
    )

    """
    Configuration for Optimizer
    """

    lr: float = field(
        default = 0.001,
        metadata = {"help": "Learning rate."}
    )

    """
    Configuration for Trainer
    """

    epochs: int = field(
<<<<<<< HEAD
        default = 100,
=======
        default = 30,
>>>>>>> 1bfbd636963242e6a60651461ff7fffecef5e836
        metadata = {"help": "Number of epochs."}
    )

    display_progress: bool = field(
        default = True,
        metadata = {"help": "Display progress."}
    )

    test_interval: int = field(
        default = 1,
        metadata = {"help": "Test interval."}
    )

    """
    Configuration for Dataset.
    """

    dataset_name: Optional[str] = field(
        default = "CIFAR10", 
        metadata = {"help": "Choose ImageNet or CIFAR10 dataset."}
    )

    path_to_data: Optional[str] = field(
        default = "'./data'", 
        metadata = {"help": ""}
    )

    download_dataset: Optional[bool] = field(
        default = True,
        metadata = {"help": "Download the dataset from TorchVision."}
    )

    """
    Configuration for dataloader.
    """

    shuffle: Optional[bool] = field(
        default = True,
        metadata = {"help": "Whether to shuffle the data."}
    )

    seed: Optional[int] = field(
        default = 42, 
        metadata = {"help": "Random seed used for reproducibility."}
    )
    
    batch_size: Optional[int] = field(
        default = 256, 
        metadata = {"help": "Batch size for training and validation."}
    )

    add_sampler: Optional[bool] = field(
        default = True,
        metadata = {"help": "Whether to add DistributedDataParallelSampler to the data loader."}
    )

    drop_last: Optional[bool] = field(
        default = False,
        metadata = {"help": "Drop the last incomplete batch if True."}
    )

    pin_memory: Optional[bool] = field(
        default = False,
        metadata = {"help": "Whether to use pinned memory address in CPU memory."}
    )

    num_workers: Optional[int] = field(
        default = 0,
        metadata = {"help": "Number of worker threads for data loading."}
    )

    """
    Configuration for Weights and Biases.
    """

    use_wandb: Optional[bool] = field(
        default = False,
        metadata = {"help": "Whether to use Weights and Biases."}
    )

    project_name: Optional[str] = field(
        default = "'Vision Transformer'",
        metadata = {"help": "Name of the project."}
    )