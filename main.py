from torch.utils.data import DataLoader
from torchvision import transforms
import os

from model import ResNeXt101_64x4d
from dataset import ImageDataset, TestDataset
from utils import (
    parse_args,
    get_latest_checkpoint,
    cross_validate,
    ensemble_validate,
    ensemble_test
)

class_mapping = [
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99
]
# !!! Remember to change this for each experiment !!!
model_name = 'resnext101_64x4d_crossVal_mixup_cutmix'

if __name__ == '__main__':
    args = parse_args(model_name)

    # Set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    if args.train:
        model = ResNeXt101_64x4d(model_name)

        # Load train and validation dataset
        train_dataset = ImageDataset(
            'data/train',
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
            ])
        )
        val_dataset = ImageDataset(
            'data/val',
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        model.train(args, train_loader, val_loader, lr=1e-4, ckpt_dir='ckpt')
    elif args.test:
        model = ResNeXt101_64x4d(model_name)

        # Load test dataset
        test_dataset = TestDataset('data/test', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
        test_loader = DataLoader(test_dataset, batch_size=1)

        if args.ckpt:
            print(f'Testing with {args.ckpt}...')
            model.test(test_loader, args.ckpt, args.output_zip)
        else:
            best_ckpt = f'ckpt/{model_name}_fold1_best.pth'
            output_zip_best = args.output_zip.replace('.zip', '_best.zip')
            print("Testing with best checkpoint...")
            model.test(test_loader, best_ckpt, output_zip_best)

            last_ckpt = get_latest_checkpoint('ckpt', model_name)
            output_zip_last = args.output_zip.replace('.zip', '_last.zip')
            print("Testing with last checkpoint...")
            model.test(test_loader, last_ckpt, output_zip_last)
    elif args.train_crossVal:
        # Load train and validation dataset
        train_dataset = ImageDataset(
            'data/train',
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
            ])
        )
        val_dataset = ImageDataset(
            'data/val',
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        )
        all_dataset = train_dataset + val_dataset

        cross_validate(args, model_name, all_dataset, lr=1e-4, ckpt_dir='ckpt')
    elif args.test_ensemble:
        # Load test dataset
        test_dataset = TestDataset('data/test', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=args.num_workers
        )

        best_ckpts = [
            f'ckpt/{model_name}_fold{i+1}_best.pth'
            for i in range(args.num_folds)
        ]
        ensemble_test(
            args, model_name, test_loader, best_ckpts, args.output_zip
        )
    elif args.val_ensemble:
        # Load validation dataset
        val_dataset = ImageDataset('data/val', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]))
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers
        )

        best_ckpts = [
            f'ckpt/{model_name}_fold{i+1}_best.pth'
            for i in range(args.num_folds)
        ]
        ensemble_validate(args, model_name, val_loader, best_ckpts)
    else:
        print("Please specify --train, --test, --train_crossVal, "
              "--test_ensemble or --val_ensemble")
