# mlp_mnist.py
# ------------------------------------------------------------
# MLP (Multi-Layer Perceptron) em PyTorch para classificação no MNIST
# - Substitui a arquitetura Vision Transformer
# - Arquitetura simples com camadas lineares e ativações ReLU
# ------------------------------------------------------------
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback simples

# ============================================================
# 1) ARQUITETURA MULTI-LAYER PERCEPTRON (MLP)
# ============================================================
class MLP(nn.Module):
    """
    Arquitetura MLP padrão:
    Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear (Logits)
    """
    def __init__(self, input_size=28*28, hidden_size1=512, hidden_size2=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 1, 28, 28) - tensor da imagem de entrada
        Retorna logits (B, num_classes)
        """
        return self.net(x)

# ============================================================
# 2) TREINO / AVALIAÇÃO (Idênticos aos anteriores)
# ============================================================
def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def get_loaders(batch_size=64):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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
        x, y = x.to(device), y.to(device)

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
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits, y) * x.size(0)
        n += x.size(0)

    return total_loss / n, total_acc / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true", help="força CPU (default usa cuda se disponível)")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)

    # Instancia o modelo MLP
    model = MLP().to(device)
    print(model)

    # Contar parâmetros
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número total de parâmetros: {num_params:,}")


    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, device)
        te_loss, te_acc = eval_epoch(model, test_dl, device)
        dt = time.time() - t0
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={te_loss:.4f} acc={te_acc:.4f} | {dt:.1f}s")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "mlp_mnist_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()