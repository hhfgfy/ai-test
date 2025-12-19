# eval_preview.py
import os, glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

FRAMES_DIR = "data/frames"
LABELS_PATH = "data/labels.npy"
CKPT_PATH = "policy.pt"

IMG_SIZE = 96
OUT_DIM = 6
KEYS = ["UP","DOWN","LEFT","RIGHT","ALT","SHIFT"]

THR_DIR = 0.5   # 방향키 표시 임계값
THR_TAP = 0.6   # 탭키(ALT/SHIFT) 표시 임계값(조금 보수적으로)

class ResNet18Policy(nn.Module):
    def __init__(self, out_dim=6):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)  # logits

    def forward(self, x):
        return self.backbone(x)

def preprocess(path):
    img = cv2.imread(path)  # BGR
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    return x

def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    model = ResNet18Policy(out_dim=OUT_DIM)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    labels = np.load(LABELS_PATH)  # (N,6)

    n = min(len(frame_paths), len(labels))
    frame_paths = frame_paths[:n]
    labels = labels[:n]

    for i, p in enumerate(frame_paths):
        x = preprocess(p).unsqueeze(0)  # (1,3,H,W)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)[0].numpy()  # (6,)

        pred = np.zeros((OUT_DIM,), dtype=np.uint8)
        pred[:4] = (prob[:4] >= THR_DIR).astype(np.uint8)
        pred[4:] = (prob[4:] >= THR_TAP).astype(np.uint8)

        gt = labels[i]

        # 원본 프레임(저장된 jpg)로 보기
        img_show = cv2.imread(p)

        text1 = "PRED " + " ".join([f"{k}:{int(pred[j])}({prob[j]:.2f})" for j,k in enumerate(KEYS)])
        text2 = "GT   " + " ".join([f"{k}:{int(gt[j])}" for j,k in enumerate(KEYS)])

        cv2.putText(img_show, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.putText(img_show, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(img_show, "q: quit", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        cv2.imshow("Eval Preview (ResNet18, single frame, no input)", img_show)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
