from pathlib import Path

import torch
from dataset import Dataset
from metrics import AddMarginProduct
from models import resnet_face18
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from tqdm import tqdm

NUM_CLASSES: int = 50
DATASET_DIR: str = 'C:/Users/User/Documents/Project/face_detector_from_scratch/datasets/celeba-recog-50'
BATCH_SIZE: int = 8
NUM_WORKERS: int = 0
DEVICE: str = 'cpu'
EPOCH: int = 100


device = torch.device(DEVICE)
data_dir = Path(DATASET_DIR)
train_dataset = Dataset(data_dir / 'train', phase='train')
trainloader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


criterion = nn.CrossEntropyLoss()

model = resnet_face18()

metric_fc = AddMarginProduct(512, NUM_CLASSES, s=30, m=0.35)

optimizer = SGD(
    [{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    lr=0.1, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

@torch.no_grad
def count_correct(output, target) -> int:
    _, predictions = torch.max(output, dim=1)
    return torch.sum(predictions == target).item()

for epoch_i in range(EPOCH):
    scheduler.step()
    model.train()
    epoch_loss = 0.
    epoch_acc = 0.

    dataloader = tqdm(trainloader, total=len(trainloader), leave=False)
    for batch_i, batch_data in enumerate(dataloader):
        images, labels = batch_data
        images = images.to(device)
        labels = labels.to(device).long()
        
        embedding = model(images)
        output = metric_fc(embedding, labels)

        loss: torch.Tensor = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += count_correct(output, labels)
    
    epoch_loss /= len(train_dataset)
    epoch_acc /= len(train_dataset)
    print(f'Epoch {epoch_i} | loss {epoch_loss:.2e} | acc {epoch_acc:.0%}')