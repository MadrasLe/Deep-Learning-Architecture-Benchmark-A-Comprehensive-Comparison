# unet_classifier.py
# ------------------------------------------------------------
# Usa o Encoder de uma U-Net pré-treinada em denoising como
# a base para um classificador de imagens.
# Fase 1: Treina a U-Net em remoção de ruído.
# Fase 2: Congela o encoder, adiciona uma cabeça de classificação
#         e treina apenas a nova cabeça.
# ------------------------------------------------------------
import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x

# ============================================================
# 1) ARQUITETURA U-NET (A mesma de antes)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        return torch.sigmoid(self.outc(x))

# ============================================================
# 2) O NOVO MODELO: U-NET CLASSIFICADOR
# ============================================================
class UNetClassifier(nn.Module):
    def __init__(self, trained_unet, num_classes=10):
        super().__init__()
        # Pega o encoder da U-Net treinada
        self.encoder = nn.Sequential(
            trained_unet.inc,
            trained_unet.down1,
            trained_unet.down2
        )
        # Adiciona a cabeça de classificação
        # A saída do encoder é (B, 256, 7, 7)
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Reduz cada feature map para 1x1
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier_head(features)

# ============================================================
# 3) LÓGICA DE TREINO E AVALIAÇÃO
# ============================================================
def get_loaders(batch_size=64):
    # Para o treino do classificador, normalizar é uma boa prática
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    )

def train_classifier_epoch(model, dl, opt, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="Classifier Train"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).float().sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_classifier_epoch(model, dl, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="Classifier Eval"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).float().sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--denoise_epochs", type=int, default=5)
    p.add_argument("--classifier_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise_factor", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")
    train_dl, test_dl = get_loaders(args.batch_size)

    # --- FASE 1: PRÉ-TREINO DA U-NET EM DENOISING ---
    print("\n--- FASE 1: PRÉ-TREINANDO A U-NET ---")
    unet_model = UNet().to(device)
    unet_opt = torch.optim.Adam(unet_model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.denoise_epochs + 1):
        unet_model.train()
        for clean_imgs, _ in tqdm(train_dl, desc=f"Denoise Epoch {epoch}"):
            clean_imgs = clean_imgs.to(device)
            noisy_imgs = torch.clamp(clean_imgs + args.noise_factor * torch.randn_like(clean_imgs), -1., 1.)
            unet_opt.zero_grad()
            denoised_imgs = unet_model(noisy_imgs)
            loss = loss_fn(denoised_imgs, clean_imgs)
            loss.backward()
            unet_opt.step()
        print(f"Denoise Epoch {epoch} concluída.")
    
    # --- FASE 2: TREINO SUPERVISIONADO DO CLASSIFICADOR ---
    print("\n--- FASE 2: TREINANDO O CLASSIFICADOR ---")
    classifier_model = UNetClassifier(unet_model).to(device)
    
    # CONGELA o encoder pré-treinado
    for param in classifier_model.encoder.parameters():
        param.requires_grad = False
        
    # Otimizador para treinar apenas a nova cabeça
    classifier_opt = torch.optim.Adam(classifier_model.classifier_head.parameters(), lr=args.lr)
    
    best_acc = 0.0
    for epoch in range(1, args.classifier_epochs + 1):
        tr_loss, tr_acc = train_classifier_epoch(classifier_model, train_dl, classifier_opt, device)
        te_loss, te_acc = eval_classifier_epoch(classifier_model, test_dl, device)
        print(f"[Classifier Epoch {epoch}] Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Test Loss={te_loss:.4f} Acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(classifier_model.state_dict(), "unet_classifier_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()