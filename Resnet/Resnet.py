# resnet_micro.py
# ------------------------------------------------------------
# Uma ResNet realmente "micro" com um número de parâmetros muito menor.
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
    tqdm = lambda x, **k: x

# ResidualBlock não precisa de nenhuma mudança, a lógica é a mesma.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class MicroResNet(nn.Module):
    def __init__(self, block=ResidualBlock, num_blocks=[2, 2, 2], num_classes=10):
        super().__init__()
        # Começamos com um número de canais bem menor
        self.in_channels = 16  # <-- MUDANÇA (era 64)

        # Camada inicial
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False), # <-- MUDANÇA (era 64)
            nn.BatchNorm2d(16), # <-- MUDANÇA (era 64)
            nn.ReLU(inplace=True)
        )

        # Stacking dos blocos com canais reduzidos
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # <-- MUDANÇA (era 64)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # <-- MUDANÇA (era 128)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # <-- MUDANÇA (era 256)

        # Classificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes) # <-- MUDANÇA (era 256)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# ============================================================
# O restante do código (treino, avaliação, main) é EXATAMENTE O MESMO
# Basta mudar a linha que instancia o modelo na função main.
# ============================================================

def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def get_loaders(batch_size=128):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl

def train_epoch(model, dl, opt, device, grad_clip=1.0):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="train"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
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
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)

    # Instancia o NOVO modelo MicroResNet
    model = MicroResNet(num_blocks=[2, 2, 2]).to(device)
    print(model)

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
            torch.save(model.state_dict(), "resnet_micro_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()