import torch
import torchvision
import torchvision.transforms as transforms

from base import BaseDataLoader

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2023, 0.1994, 0.2010])
])

def get_dataloader(
    data_name: str = 'cifar100',
    data_augmentation=True,
    batch_size: int = 128,
    num_workers: int = 2
    ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    Get dataloader for the required dataset
    Args:
        data_name (str, optional): Name of the dataset. Defaults to 'cifar100'.
        data_augmentation (bool, optional): Whether to apply data augmentation. Defaults to True.
        batch_size (int, optional): Number of samples in a batch. Defaults to 128.
        num_workers (int, optional): Number of workers to use for data loading. Defaults to 2.
    """
    train_dataset, test_dataset = None, None
    if data_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
    elif data_name == 'imagenet':
        # train_dataset
    
    train_loader = BaseDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = BaseDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_dataloader()
    print(f'Train loader: {len(train_loader)} batches')
    print(f'Test loader: {len(test_loader)} batches')
    # print(f'Validation loader: {len(val_loader)} batches')
    print(f'Number of classes: {train_loader.dataset} classes')


