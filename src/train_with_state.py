import os, glob, random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

FRAMES_DIR = "data/frames"
LABELS_PATH = "data/labels.npy"
STATE_PATH  = "data/state.npy"
OUT_PATH = "policy_img_state.pt"

IMG_SIZE = 96
BATCH = 32
EPOCHS = 10
LR = 1e-3
VAL_RATIO = 0.2
SEED = 42

KEY_DIM = 6
STATE_DIM = 6  # [px,py,mx,my,dx,dy]

class ImitationDataset(Dataset):
    def __init__(self, frame_paths, labels, states):
        self.frame_paths = frame_paths
        self.labels = labels
        self.states = states

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.frame_paths[idx])  # BGR
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        x_state = torch.from_numpy(self.states[idx]).float()            # (6,)
        y = torch.from_numpy(self.labels[idx]).float()                  # (6,)
        return x_img, x_state, y

class ImgStatePolicy(nn.Module):
    def __init__(self, out_dim=6, state_dim=6):
        super().__init__()

        # ResNet18 backbone: 마지막 fc를 feature extractor로 바꿈
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone  # output: (feat_dim,)

        # state MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # head
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 64, 128), nn.ReLU(),
            nn.Linear(128, out_dim)  # logits
        )

    def forward(self, x_img, x_state):
        f_img = self.backbone(x_img)        # (B, feat_dim)
        f_state = self.state_mlp(x_state)   # (B, 64)
        f = torch.cat([f_img, f_state], dim=1)
        return self.head(f)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    labels = np.load(LABELS_PATH)   # (N,6)
    states = np.load(STATE_PATH)    # (N,6)

    n = min(len(frame_paths), len(labels), len(states))
    frame_paths = frame_paths[:n]
    labels = labels[:n]
    states = states[:n]

    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(n * (1 - VAL_RATIO))
    train_idx, val_idx = idxs[:split], idxs[split:]

    train_ds = ImitationDataset([frame_paths[i] for i in train_idx], labels[train_idx], states[train_idx])
    val_ds   = ImitationDataset([frame_paths[i] for i in val_idx],   labels[val_idx],   states[val_idx])

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImgStatePolicy(out_dim=KEY_DIM, state_dim=STATE_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"N={n} | train={len(train_ds)} val={len(val_ds)} | device={device}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for x_img, x_state, y in train_dl:
            x_img, x_state, y = x_img.to(device), x_state.to(device), y.to(device)
            logits = model(x_img, x_state)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y.size(0)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x_img, x_state, y in val_dl:
                x_img, x_state, y = x_img.to(device), x_state.to(device), y.to(device)
                logits = model(x_img, x_state)
                loss = loss_fn(logits, y)
                va_loss += loss.item() * y.size(0)

        tr_loss /= max(1, len(train_ds))
        va_loss /= max(1, len(val_ds))
        print(f"epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

    torch.save(
        {"model_state": model.state_dict(), "img_size": IMG_SIZE, "uses_state": True},
        OUT_PATH
    )
    print("saved:", OUT_PATH)

if __name__ == "__main__":
    main()
