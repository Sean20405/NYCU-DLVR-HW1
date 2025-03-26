import torch
from torch import nn, optim
from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

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
        self.model_name = model_name
        self.writer = SummaryWriter(f'runs/exp/{self.model_name}')
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
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img)
        return pred

    def train(self, args, train_loader, val_loader, lr=1e-4, ckpt_dir='ckpt'):
        os.makedirs(ckpt_dir, exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        best_val_acc = 0.0
        total_training_time = 0.0

        cutmix = v2.CutMix(num_classes=self.num_class)
        mixup = v2.MixUp(num_classes=self.num_class)
        cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])

        # Log hyperparameters
        self.writer.add_text(
            'Parameters',
            f'Learning Rate: {lr}, '
            f'Epochs: {args.epochs}, '
            f'Batch Size: {train_loader.batch_size}'
        )

        for epoch in range(args.epochs):
            epoch_start_time = time.time()

            # Training
            self.model.train()
            for i, (img, label) in enumerate(train_loader):  # For each batch
                img, label = img.to(self.device), label.to(self.device)
                img, label = cutmix_or_mixup(img, label)
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
                checkpoint_path = f'{ckpt_dir}/{self.model_name}_epoch' \
                                  f'{epoch + 1}.pth'
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
                best_model_path = f'{ckpt_dir}/{self.model_name}_best.pth'
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
                pred_label = class_mapping[int(pred_label.item())]
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
