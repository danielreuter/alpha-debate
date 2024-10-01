# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
from dataclasses import dataclass
import wandb
from tqdm import tqdm

@dataclass
class ConvNetArgs:
    num_pixels: int = 6
    input_channels: int = 2
    conv_out_channels: int = 32
    fc_out_features: int = 128
    num_classes: int = 10

@dataclass
class ConvNetTrainerArgs:
    batch_size: int = 128
    lr: float = 1e-4
    num_batches: int = 30000  # Will be set to 50000 for 4 pixels
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "mnist-sparse"
    wandb_name: str = "conv-net"

class ConvNet(nn.Module):
    def __init__(self, args: ConvNetArgs):
        super().__init__()
        self.args = args
        self.conv = nn.Conv2d(args.input_channels, args.conv_out_channels, 3, padding=1)
        self.fc1 = nn.Linear(28 * 28 * args.conv_out_channels, args.fc_out_features)
        self.fc2 = nn.Linear(args.fc_out_features, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MaskAllButNPixels(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, img):
        """
        Apply mask to the image, revealing only `n` random non-zero pixels.

        Args:
            img (torch.Tensor): Image tensor with shape (1, 28, 28).

        Returns:
            torch.Tensor: Masked image tensor with shape (2, 28, 28).
        """
        mask = torch.zeros_like(img)
        # Get indices of non-zero pixels
        non_zero_indices = torch.nonzero(img[0] > 0, as_tuple=False)
        non_zero_indices = [tuple(idx.tolist()) for idx in non_zero_indices]
        if len(non_zero_indices) >= self.n:
            pixels = random.sample(non_zero_indices, self.n)
        else:
            pixels = non_zero_indices  # Use all non-zero pixels if fewer than n
        for i, j in pixels:
            mask[0, i, j] = 1  # Mask channel indicating revealed pixels
        masked_img = torch.cat([mask, img * mask], dim=0)
        return masked_img

class DynamicMaskDataset(Dataset):
    def __init__(self, dataset, num_pixels):
        self.dataset = dataset
        self.num_pixels = num_pixels
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
        ])
        self.mask_transform = MaskAllButNPixels(num_pixels)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.transform(img)  # Now img is a tensor in [0, 1] range
        masked_img = self.mask_transform(img)
        return masked_img, label

    def __len__(self):
        return len(self.dataset)

class ConvNetTrainer:
    def __init__(self, model: ConvNet, args: ConvNetTrainerArgs):
        self.model = model
        self.args = args
        self.device = torch.device(args.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # No normalization here, to match Keras code
        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        self.trainset = DynamicMaskDataset(mnist_dataset, model.args.num_pixels)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)

        mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
        self.testset = DynamicMaskDataset(mnist_testset, model.args.num_pixels)
        self.testloader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)

    def train(self):
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=vars(self.args))
        wandb.watch(self.model)
        
        try:
            pbar = tqdm(total=self.args.num_batches, desc="Training")
            batch_count = 0
            while batch_count < self.args.num_batches:
                self.model.train()
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    wandb.log({
                        "loss": loss.item(),
                    }, step=batch_count)
                    
                    batch_count += 1
                    pbar.update(1)
                    
                    if batch_count % 100 == 0:
                        accuracy = self.evaluate()
                        wandb.log({
                            "accuracy": accuracy,
                        }, step=batch_count)
                    
                    if batch_count >= self.args.num_batches:
                        break
            
            pbar.close()
        finally:    
            wandb.finish()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

import os 

def get_judge(num_pixels: int, model_path: str):
    conv_net_args = ConvNetArgs(
        num_pixels=num_pixels,
        conv_out_channels=32,
        fc_out_features=128,
    )
    judge = ConvNet(conv_net_args)
    if os.path.exists(model_path):
        judge.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        judge.eval()
        return judge
    else:
        raise ValueError(f"Judge model not found at {model_path}")
    
if __name__ == "__main__":
    
    num_pixels = 6
    judge = get_judge(num_pixels)

    trainer_args = ConvNetTrainerArgs(
        batch_size=128,
        lr=1e-4,
        num_batches=30000 if num_pixels == 6 else 50000,
    )
    trainer = ConvNetTrainer(judge, trainer_args)

    # Training
    trainer.train()

    # Final evaluation
    final_accuracy = trainer.evaluate()
    print(f'Final Accuracy of the CNN on the test images ({num_pixels} pixels): {final_accuracy:.2f}%')

    # Save the judge
    torch.save(judge.state_dict(), f"{num_pixels}_pixel_judge.pth")

# %%
