# cnn_mnist.py
# ------------------------------------------------------------
# CNN tradicional em PyTorch puro para classificação no MNIST
# - Substitui a SNN do código original
# - Arquitetura convolucional padrão com ReLU e MaxPool
# - Treina em CPU (ou GPU se disponível)
# - Dataset: MNIST
# ------------------------------------------------------------
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback simples

# ============================================================
# 1) ARQUITETURA CNN (Substituindo a SNN)
# ============================================================
class CNN(nn.Module):
    """
    Arquitetura CNN padrão:
    Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear -> ReLU -> Linear (Logits)
    Processa a imagem de entrada em um único passo (forward pass).
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Bloco 1: (B, 1, 28, 28) -> (B, 16, 14, 14)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Bloco 2: (B, 16, 14, 14) -> (B, 32, 7, 7)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classificador: (B, 32*7*7) -> (B, 10)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Inicialização dos pesos
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 1, 28, 28) - tensor da imagem de entrada
        Retorna logits (B, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        logits = self.classifier(x)
        return logits

# ============================================================
# 2) TREINO / AVALIAÇÃO (Simplificados para CNN)
# ============================================================
def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def get_loaders(batch_size=64):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Normalização padrão para MNIST
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl

def train_epoch(model, dl, opt, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()

    for x, y in tqdm(dl, desc="train"):
        x = x.to(device)   # (B,1,28,28)
        y = y.to(device)   # (B,)

        # Forward pass direto com a imagem
        logits = model(x)
        loss = ce(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits, y) * x.size(0)
        n += x.size(0)

    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in tqdm(dl, desc="eval"):
        x = x.to(device)
        y = y.to(device)

        # Forward pass direto
        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits, y) * x.size(0)
        n += x.size(0)

    return total_loss / n, total_acc / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true", help="força CPU (default usa cuda se disponível)")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)
    
    # Instancia o modelo CNN
    model = CNN().to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Funções de treino/avaliação não precisam mais do encoder ou do T
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, device)
        te_loss, te_acc = eval_epoch(model, test_dl, device)
        dt = time.time() - t0
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={te_loss:.4f} acc={te_acc:.4f} | {dt:.1f}s")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "cnn_mnist_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()