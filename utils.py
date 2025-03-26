import torch
from torch.utils.data import DataLoader, Subset

import os
import argparse
import pandas as pd
from PIL import Image
import tqdm
import zipfile
from sklearn.model_selection import KFold
import numpy as np

from model import ResNeXt101_64x4d

class_mapping = [
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99
]


def parse_args(model_name):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a model without cross validation'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test for a single model'
    )
    parser.add_argument(
        '--train_crossVal',
        action='store_true',
        help='Train the model with cross validation'
    )
    parser.add_argument(
        '--test_ensemble',
        action='store_true',
        help='Test the model with ensemble'
    )
    parser.add_argument(
        '--val_ensemble',
        action='store_true',
        help='Validate the model with ensemble'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU ID'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for DataLoader'
    )
    parser.add_argument(
        '--freq_save_ckpt',
        type=int,
        default=20,
        help='Frequency to save checkpoint'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=80,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_folds',
        type=int,
        default=5,
        help='Number of folds for cross validation'
    )
    parser.add_argument(
        '--fold_begin',
        type=int,
        default=0,
        help='Fold number to begin training'
    )
    parser.add_argument(
        '--fold_end',
        type=int,
        default=5,
        help='Fold number to end training'
    )
    parser.add_argument(
        '--output_zip',
        type=str,
        default=f'result/{model_name}.zip',
        help='Output ZIP file for test results'
    )

    return parser.parse_args()


def get_latest_checkpoint(ckpt_dir, model_name):
    checkpoints = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(model_name) and f.endswith('.pth')
    ]
    checkpoints.sort(
        key=lambda x: int(x.split('_epoch')[1].split('.pth')[0])
        if '_epoch' in x else -1
    )
    return os.path.join(ckpt_dir, checkpoints[-1]) if checkpoints else None


def cross_validate(args, model_name, dataset, lr=1e-4, ckpt_dir='ckpt'):
    print(f'Total number of samples: {len(dataset)}')
    os.makedirs(ckpt_dir, exist_ok=True)

    kfold = KFold(n_splits=args.num_folds, shuffle=True)

    # For each fold, [fold_begin, fold_end)
    kfold_splits = kfold.split(np.arange(len(dataset)))
    for fold, (train_idx, val_idx) in enumerate(kfold_splits):  # For each fold
        if fold < args.fold_begin or fold >= args.fold_end:
            continue
        print(f'=========== Fold {fold+1}/{args.num_folds} ===========')

        model = ResNeXt101_64x4d(f'{model_name}_fold{fold+1}')

        train_subset = Subset(dataset, list(train_idx))
        val_subset = Subset(dataset, list(val_idx))

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        model.train(
            args, train_loader, val_loader, lr=lr, ckpt_dir=f'{ckpt_dir}'
        )


def ensemble_validate(args, model_name, dataloader, ckpt_paths):
    models = [ResNeXt101_64x4d(model_name) for _ in ckpt_paths]
    for model, ckpt_path in zip(models, ckpt_paths):
        model.load_checkpoint(ckpt_path)
        model.model.eval()

    acc = 0.0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(models[0].device), label.to(models[0].device)
            pred = torch.zeros(1, 100).to(models[0].device)
            for model in models:
                pred += model.model(img)
            _, pred_label = torch.max(pred, 1)

            # Calculate accuracy
            acc += (label == pred_label).item()

    acc /= dataloader.__len__()
    print(f'Validation accuracy: {acc}')


def ensemble_test(
    args, model_name, test_loader, ckpt_paths, output_zip='solution.zip'
):
    models = [
        ResNeXt101_64x4d(f'{model_name}_fold{i+1}')
        for i, _ in enumerate(ckpt_paths)
    ]
    for model, ckpt_path in zip(models, ckpt_paths):
        model.load_checkpoint(ckpt_path)
    for model in models:
        model.model.eval()

    results = []
    with torch.no_grad():
        for img_path in tqdm.tqdm(test_loader):
            img_path = img_path[0]
            img = Image.open(img_path).convert('RGB')
            img = models[0].transform(img).unsqueeze(0).to(models[0].device)

            pred = torch.zeros(1, 100).to(models[0].device)
            for model in models:
                pred += model.model(img)
            _, pred_label = torch.max(pred, 1)
            pred_label = class_mapping[int(pred_label.item())]

            base_name = os.path.basename(img_path)
            img_name = base_name[:base_name.rfind('.')]
            results.append((os.path.basename(img_name), pred_label))

    # Save results to CSV and compress to ZIP
    csv_path = output_zip.replace('.zip', '.csv')
    df = pd.DataFrame(results, columns=['image_name', 'pred_label'])
    df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(output_zip, 'w') as zf:
        zf.write(csv_path, arcname='prediction.csv')

    print(f'Results saved to {output_zip}')
