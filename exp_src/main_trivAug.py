import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import time
import pandas as pd
from PIL import Image
import tqdm
import zipfile


class_mapping = [
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99
]
# !!! Remember to change this for each experiment !!!
model_name = 'resnext101_64x4d_trivAug'


class ResNeXt101_64x4d:
    def __init__(self, model_name):
        super(ResNeXt101_64x4d, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.num_class = 100
        self.model = resnext101_64x4d(
            weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.writer = SummaryWriter(f'runs/exp/{model_name}')
        model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f'Model size: {model_size:.2f}M parameters')

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f'Model loaded from {ckpt_path}')
        print(f'Epoch: {ckpt['epoch']}, '
              f'Validation Accuracy: {ckpt['val_acc']:.4f}')
        return ckpt

    def predict(self, img):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img)
        return pred

    def train(self, args, train_loader, val_loader, lr=1e-3, ckpt_dir='ckpt'):
        os.makedirs(ckpt_dir, exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        best_val_acc = 0.0
        total_training_time = 0.0

        # Log hyperparameters
        self.writer.add_text(
            'Parameters',
            f'Learning Rate: {lr}, Epochs: {args.epochs}, '
            f'Batch Size: {train_loader.batch_size}'
        )

        for epoch in range(args.epochs):
            epoch_start_time = time.time()

            # Training
            self.model.train()
            for i, (img, label) in enumerate(train_loader):  # For each batch
                img, label = img.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                pred = self.model(img)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch {epoch}, Iter {i}, Loss: {loss.item()}')
                    self.writer.add_scalar(
                        'training loss',
                        loss.item(),
                        epoch * len(train_loader) + i
                    )
                    self.writer.add_scalar(
                        'learning rate',
                        optimizer.param_groups[0]['lr'],
                        epoch * len(train_loader) + i
                    )

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for img, label in val_loader:
                    img, label = img.to(self.device), label.to(self.device)
                    pred = self.model(img)
                    _, pred = torch.max(pred, 1)
                    correct += (pred == label).sum().item()
                    total += label.size(0)

            val_acc = correct / total

            # Calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            total_training_time += epoch_time
            self.writer.add_scalar('epoch_time', epoch_time, epoch)

            print(f'Epoch {epoch}, Val Acc: {val_acc:.4f}, '
                  f'Time: {epoch_time:.2f} sec')
            self.writer.add_scalar('validation accuracy', val_acc, epoch)

            # Save checkpoint
            if (epoch + 1) % args.freq_save_ckpt == 0:
                checkpoint_path = (
                    f'{ckpt_dir}/{model_name}_epoch{epoch + 1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'loss': loss.item(),
                    'epoch_time': epoch_time,
                    'total_training_time': total_training_time,
                }, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = f'{ckpt_dir}/{model_name}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'loss': loss.item(),
                    'epoch_time': epoch_time,
                    'total_training_time': total_training_time,
                }, best_model_path)
                print(f'Best model saved to {best_model_path}')

        # Print total training time
        hours = int(total_training_time / 3600)
        minutes = int((total_training_time % 3600) / 60)
        seconds = int(total_training_time % 60)
        print(f'Total training time: {hours}h {minutes}m {seconds}s')

        self.writer.close()

    def test(self, test_loader, ckpt_path, output_zip='solution.zip'):
        self.load_checkpoint(ckpt_path)
        self.model.eval()
        results = []
        with torch.no_grad():
            for img_path in tqdm.tqdm(test_loader):
                img_path = img_path[0]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img).unsqueeze(0).to(self.device)
                pred = self.model(img)
                _, pred_label = torch.max(pred, 1)
                pred_label = class_mapping[pred_label.item()]
                img_name = os.path.basename(img_path)
                img_name = img_name[:img_name.rfind('.')]
                results.append((os.path.basename(img_name), pred_label))

        # Save results to CSV and compress to ZIP
        csv_path = output_zip.replace('.zip', '.csv')
        df = pd.DataFrame(results, columns=['image_name', 'pred_label'])
        df.to_csv(csv_path, index=False)
        with zipfile.ZipFile(output_zip, 'w') as zf:
            zf.write(csv_path, arcname='prediction.csv')

        print(f'Results saved to {output_zip}')


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = datasets.ImageFolder(root, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [
            os.path.join(root, img) for img in os.listdir(root)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        return img_path


def get_latest_checkpoint(ckpt_dir, model_name):
    checkpoints = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(model_name) and f.endswith('.pth')
    ]
    checkpoints.sort(
        key=lambda x: int(x.split('_epoch')[1].split('.pth')[0])
        if '_epoch' in x else -1
    )
    checkpoints.sort(
        key=lambda x: int(x.split('_epoch')[1].split('.pth')[0])
        if '_epoch' in x else -1
    )
    return os.path.join(ckpt_dir, checkpoints[-1]) if checkpoints else None


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the model'
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
        default=100,
        help='Number of epochs for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--output_zip',
        type=str,
        default=f'result/{model_name}.zip',
        help='Output ZIP file for test results'
    )

    args = parser.parse_args()

    # Set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    model = ResNeXt101_64x4d(model_name)

    if args.train:
        # Load train and validation dataset
        train_dataset = ImageDataset(
            'data/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
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
            batch_size=args.batch_size
        )

        model.train(args, train_loader, val_loader, lr=1e-4, ckpt_dir='ckpt')
    if args.test:
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
            best_ckpt = f'ckpt/{model_name}_best.pth'
            last_ckpt = get_latest_checkpoint('ckpt', model_name)
            print('Testing with best checkpoint...')
            model.test(
                test_loader,
                best_ckpt,
                args.output_zip.replace('.zip', '_best.zip')
            )
            print('Testing with last checkpoint...')
            model.test(
                test_loader,
                last_ckpt,
                args.output_zip.replace('.zip', '_last.zip')
            )
