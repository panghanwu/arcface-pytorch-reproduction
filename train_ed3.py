from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import cosine_similarity, normalize
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceRecognitionData
from metrics import ArcFaceHead, ArcFaceLoss
from models import create_mobilenet_large_for_classification
from utils import create_log_dir, set_seed
from painter import plot_3d_embeddings

TITLE: str = 'emb3-16'
DATASET_DIR: str = 'C:/Users/User/Documents/Project/face_detector_from_scratch/datasets/celeba-recog-16'
BATCH_SIZE: int = 8
NUM_WORKERS: int = 0
EMBEDDING_DIM: int = 3
MARGIN: int = 0.01
SCALE: int = 1.0
DEVICE: str = 'cpu'
EPOCH: int = 100

set_seed(66666666)
device = torch.device(DEVICE)
data_dir = Path(DATASET_DIR)
train_dataset = FaceRecognitionData(data_dir / 'train', augmentation=False)
trainloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

model = create_mobilenet_large_for_classification(EMBEDDING_DIM).to(device)
arcface = ArcFaceHead(train_dataset.num_classes, EMBEDDING_DIM).to(device)
criterion = ArcFaceLoss(train_dataset.num_classes, margin=MARGIN, scale=SCALE)

optimizer = Adam(
    [{'params': model.parameters()}, {'params': arcface.parameters()}],
    lr=0.001
)

log_dir = create_log_dir(TITLE)
_ = create_log_dir((log_dir/'emb'))

@torch.no_grad
def count_correct(output, target) -> int:
    _, predictions = torch.max(output, dim=1)
    return torch.sum(predictions == target).item()

def write_log(filename: str, line: str) -> None:
    with open(filename, 'a') as file:
        file.write(line + '\n')

loss_log = []
for epoch_i in range(EPOCH):
    model.train()
    epoch_loss = 0.
    epoch_acc = 0.
    embeddings = []
    groundtruth = []

    dataloader = tqdm(trainloader, total=len(trainloader), leave=False)
    for batch_i, batch_data in enumerate(dataloader):
        images, labels = batch_data
        images = images.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()

        embedding = model(images)
        output = arcface(embedding)

        loss: torch.Tensor = criterion(output, labels) + 1e-2 * arcface.loss()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += count_correct(output, labels)

        embeddings.append(embedding.detach().cpu())
        groundtruth.append(labels.detach().cpu())

    embeddings = torch.concatenate(embeddings, dim=0)
    norm_embeddings = normalize(embeddings, dim=1)
    norm_centers = normalize(arcface.weight.detach().cpu(), dim=1)
    groundtruth = torch.concatenate(groundtruth)

    plot_3d_embeddings(norm_embeddings, groundtruth, norm_centers, save_path=str((log_dir/'emb')/f'emb-{epoch_i}.png'), show=False)

    epoch_loss /= len(trainloader)
    loss_log.append(epoch_loss)
    epoch_acc /= len(train_dataset)
    lr = optimizer.param_groups[0]['lr']
    info = f'Epoch {epoch_i} | lr {lr:.1e} | loss {epoch_loss:.2e} | acc {epoch_acc:.0%}'
    write_log(log_dir / 'log.txt', info)
    print(info)

ckp = {'mb': model.cpu(), 'arc': arcface.cpu()}
torch.save(ckp, log_dir / 'checkpoint.pt')

plt.figure()
plt.plot(loss_log)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(linestyle='--', alpha=0.3)
plt.savefig(log_dir / 'loss.png')
plt.close()


# Radian matrix
vectors = norm_centers
n_vec = len(vectors)
cos_matrix = torch.full((n_vec, n_vec), fill_value=torch.nan)

for i in range(n_vec):
    cos_matrix[i:n_vec, i] = cosine_similarity(vectors[i], vectors[i:n_vec])

radian_matrix = torch.arccos(torch.clamp(cos_matrix, min=-1., max=1.))

fig, ax = plt.subplots()
ax.matshow(radian_matrix)
plt.title('Radian Matrix')
for (i, j), z in np.ndenumerate(radian_matrix):
    ax.text(j, i, f'{z:0.1f}', ha='center', va='center')

plt.savefig(log_dir / f'radian_matrix-{i}.png')
plt.close()