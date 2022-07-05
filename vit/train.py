import torch
from torch import nn, optim

import colossalai
from colossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.nn.metric import Accuracy
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.logging import get_dist_logger, disable_existing_loggers

from config import CFG

from build_dataloader import build_dataloader

from vit import vit_32_224

import bentoml

import wandb

def ViT_Trainer(args: CFG):
    assert torch.cuda.is_available()
    disable_existing_loggers()

    parser = colossalai.get_default_parser()

    parser.add_argument(
        '--use_trainer', 
        action='store_true', 
        help='whether to use trainer'
        )

    if args.use_zero == True:
        colossalai.launch_from_torch(config='./zero_config.py')

        # ViT model with ZeRO Offloading
        with ZeroInitContext(
            target_device = torch.cuda.current_device(),
            shard_strategy = gpc.config.zero.model_config.shard_strategy,
            shard_param = True
        ):
            model = vit_32_224()

    elif args.use_fp16 == True:
        colossalai.launch_from_torch(config='./fp16_config.py')

        # ViT model with fp16 
        model = vit_32_224()

    elif args.use_3d_TP == True:
        colossalai.launch_from_torch(config='./3d_config.py')

        # ViT model with 3d Tensor Parallelism
        model = vit_32_224()

    else:
        colossalai.launch_from_torch(config='./colossal_config.py')

        # ViT model 
        model = vit_32_224()

    logger = get_dist_logger()

    # setup dataloaders
    train_dataloader, test_dataloader = build_dataloader(args)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer function
    if args.use_fp16 or args.use_zero == True:
        optimizer = colossalai.nn.Lamb(
            model.parameters(), 
            lr = 1.8e-2, 
            weight_decay = 0.1
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), 
            lr = args.lr
        )

    # initialize model, optimizer, loss function, and data loaders
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, 
        optimizer, 
        loss_fn, 
        train_dataloader,
        test_dataloader
    )

    if args.use_wandb == True:

        wandb.init(project = args.project_name)

        # Training Loop
        for _ in range(args.epochs):
            engine.train()
            for step, (image, label) in train_dataloader:
                image, label = image.cuda(), label.cuda()

                engine.zero_grad()
                output = engine(image)

                train_loss = engine.loss_fn(output, label)
                wandb.log({'loss': train_loss})

                engine.backward(train_loss)
                engine.step()
                wandb.log({'step': step})
            
            # Validation Loop
            engine.eval()
            for image, label in test_dataloader:
                image, label = image.cuda(), label.cuda()

                with torch.no_grad():
                    output = engine(image)
                    test_loss = engine.loss_fn(output, label)
                    wandb.log({'test_loss': test_loss})

                engine.backward(test_loss)
                engine.step()
        
        wandb.alert(
            title = 'Training Complete',
            text = "Training complete."
        )

    else:

        # Time session
        timer = MultiTimer()

        # Trainer
        trainer = Trainer(
            engine = engine, 
            timer = timer,
            logger = logger
        )

        # hooks
        hook_list = [
            hooks.LossHook(),
            hooks.AccuracyHook(accuracy_func = Accuracy()),
            hooks.LogMetricByEpochHook(logger)
        ]

        # Training Loop
        trainer.fit(
            train_dataloader = train_dataloader,
            epochs = args.epochs, 
            test_dataloader = test_dataloader,
            hooks = hook_list,
            display_progress = args.display_progress,
            test_interval = args.test_interval
        )

    # save model
    bentoml.pytorch.save_model("vit", model)

if __name__ == "__main__":

    args = CFG()

    ViT_Trainer(args)