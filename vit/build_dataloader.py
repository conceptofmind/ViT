from torchvision.datasets import CIFAR10, ImageNet
from torchvision import transforms as T

from colossalai.utils import get_dataloader

from config import CFG

def build_dataloader(args: CFG):

    if args.dataset_name == "CIFAR10":
        policy = T.AutoAugmentPolicy.CIFAR10

    elif args.dataset_name == "ImageNet":
        policy = T.AutoAugmentPolicy.IMAGENET

    train_transform = T.Compose([
            T.Resize(args.image_size),
            T.AutoAugment(policy = policy),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = T.Compose([
            T.Resize(args.image_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])    

    if args.dataset_name  == "CIFAR10":

        train_dataset = CIFAR10(
            root = args.path_to_data, 
            train = True,
            download = args.download_dataset,
            transform = train_transform,
        )
        
        test_dataset = CIFAR10(
            root = args.path_to_data,
            train = False,
            download = args.download_dataset,
            transform = test_transform,
        )
    elif args.dataset_name == "ImageNet":

        train_dataset = ImageNet(
            root = args.path_to_data,
            train = True,
            download = args.download_dataset,
            transform = train_transform,
        )
        
        test_dataset = ImageNet(
            root = args.path_to_data,
            train = False,
            download = args.download_dataset,
            transform = test_transform,
        )

    train_loader = get_dataloader(
        train_dataset, 
        shuffle = args.shuffle,
        batch_size = args.batch_size, 
        seed = args.seed, 
        add_sampler = args.add_sampler, 
        drop_last = args.drop_last, 
        pin_memory = args.pin_memory, 
        num_workers = args.num_workers
    )
    
    test_loader = get_dataloader(
        test_dataset, 
        batch_size = args.batch_size, 
        seed = args.seed, 
        add_sampler = False, 
        drop_last = args.drop_last, 
        pin_memory = args.pin_memory, 
        num_workers = args.num_workers
    )

    return train_loader, test_loader

if __name__ == "__main__":

    train_loader, test_loader = build_dataloader(CFG)

    for batch in train_loader:
        print(batch)
        break

    for batch in test_loader:
        print(batch)
        break