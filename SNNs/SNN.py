# snn_mnist_lif.py
# ------------------------------------------------------------
# Spiking NN (LIF) em PyTorch puro com Surrogate Gradient
# - Encoder Poisson (rate coding)
# - LIF discreto com STE p/ disparo
# - Treina em CPU (ou GPU se disponível)
# - Dataset: MNIST
# ------------------------------------------------------------
import math
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
except:
    tqdm = lambda x, **k: x  # fallback simples

# ============================================================
# 1) UTIL: SURROGATE GRADIENT PARA O DISPARO (STEP FUNCTION)
# ============================================================
class SpikeFunctionSTE(torch.autograd.Function):
    """
    Forward: limiar binário (Heaviside)
    Backward: derivada suave (sigmóide) ao redor do limiar (k controla a inclinação)
    """
    @staticmethod
    def forward(ctx, v_minus_th, k=10.0):
        ctx.save_for_backward(v_minus_th)
        ctx.k = k
        return (v_minus_th >= 0).to(v_minus_th.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v_minus_th,) = ctx.saved_tensors
        k = ctx.k
        # derivada approx da heaviside: sigmoid'(k*x) = k * sigmoid(kx) * (1 - sigmoid(kx))
        sig = torch.sigmoid(k * v_minus_th)
        grad = k * sig * (1.0 - sig)
        return grad_output * grad, None

spike_fn = SpikeFunctionSTE.apply

# ============================================================
# 2) BLOCO LIF DISCRETO
# ============================================================
@dataclass
class LIFParams:
    v_th: float = 1.0        # limiar
    v_reset: float = 0.0     # reset após spike
    tau_mem: float = 20.0    # constante de tempo da membrana (passos)
    # Nota: usamos beta = exp(-1/tau_mem) como fator de "leak"

class LIFLayer(nn.Module):
    """
    LIF sem sinapses dinâmicas: integra corrente 'I' (output de uma camada linear/conv)
    v[t] = beta * v[t-1] + I[t]
    s[t] = H(v[t] - v_th)   (H é heaviside com surrogate gradient)
    Após spike: v[t] = v[t] * (1 - s[t]) + v_reset * s[t]
    """
    def __init__(self, params: LIFParams):
        super().__init__()
        self.params = params
        self.register_buffer("beta", torch.tensor(math.exp(-1.0 / params.tau_mem), dtype=torch.float32))

    def forward(self, I, v=None):
        """
        I: (B, C, H, W) ou (B, D) — corrente/sinal sináptico no passo t
        v: potencial de membrana no passo anterior (mesmas dims de I)
        Retorna: (s, v_new)
        """
        if v is None:
            v = torch.zeros_like(I)
        v = self.beta * v + I
        s = spike_fn(v - self.params.v_th)  # STE
        v = v * (1.0 - s) + self.params.v_reset * s
        return s, v

# ============================================================
# 3) ENCODER POISSON (RATE CODING)
# ============================================================
class PoissonEncoder(nn.Module):
    """
    Converte imagens [0,1] em trens de spikes Bernoulli p = intensidade
    Saída: Tensor (T, B, 1, 28, 28) com 0/1
    """
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, x):
        # x: (B, 1, 28, 28), valores em [0,1]
        B, C, H, W = x.shape
        # amostras Bernoulli independentes por timestep
        # shape: (T, B, 1, 28, 28)
        # Importante: clamp p/ estabilidade
        p = torch.clamp(x, 0.0, 1.0)
        return torch.bernoulli(p.expand(self.T, -1, -1, -1, -1))

# ============================================================
# 4) ARQUITETURA SNN
# ============================================================
class SNNConvNet(nn.Module):
    """
    Conv -> LIF -> Pool -> Conv -> LIF -> Pool -> Flatten -> Linear -> LIF -> Linear logits
    Integra ao longo de T steps com entrada codificada por spikes
    """
    def __init__(self, num_classes=10, lif_params=None):
        super().__init__()
        if lif_params is None:
            lif_params = LIFParams()

        # Bloco 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1 = LIFLayer(lif_params)
        self.pool1 = nn.MaxPool2d(2)  # 28->14

        # Bloco 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = LIFLayer(lif_params)
        self.pool2 = nn.MaxPool2d(2)  # 14->7

        # Classificador
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.lif3 = LIFLayer(lif_params)
        self.fc_out = nn.Linear(128, num_classes)

        # Inicialização leve
        for m in [self.conv1, self.conv2, self.fc1, self.fc_out]:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spikes_T):
        """
        spikes_T: (T, B, 1, 28, 28)
        Retorna logits (B, num_classes)
        Estratégia: acumulamos a taxa de disparo na penúltima camada (fc1->lif3) ao longo de T,
        depois passamos a soma média para fc_out.
        """
        T, B, _, _, _ = spikes_T.shape

        # Estados de membrana persistentes por camada
        v1 = v2 = v3 = None
        readout_acc = torch.zeros(B, 128, device=spikes_T.device)

        for t in range(T):
            x_t = spikes_T[t]  # (B,1,28,28), binário

            # Bloco 1
            I1 = self.conv1(x_t)           # (B,16,28,28)
            s1, v1 = self.lif1(I1, v1)     # spikes (B,16,28,28)
            p1 = self.pool1(s1)            # (B,16,14,14)

            # Bloco 2
            I2 = self.conv2(p1)            # (B,32,14,14)
            s2, v2 = self.lif2(I2, v2)     # (B,32,14,14)
            p2 = self.pool2(s2)            # (B,32,7,7)

            # Classificador
            flat = p2.flatten(1)           # (B,32*7*7)
            I3 = self.fc1(flat)            # (B,128)
            s3, v3 = self.lif3(I3, v3)     # (B,128)
            readout_acc = readout_acc + s3 # acumula taxa de disparo

        # Média temporal como representação
        readout_rate = readout_acc / T       # (B,128)
        logits = self.fc_out(readout_rate)   # (B,10)
        return logits

# ============================================================
# 5) TREINO / AVALIAÇÃO
# ============================================================
def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def get_loaders(batch_size=64):
    tfm = transforms.Compose([
        transforms.ToTensor(),                # [0,1]
        # opcional: normalizar p/ [0,1] já está ok
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl

def train_epoch(model, enc, dl, opt, device, T, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()

    for x, y in tqdm(dl, desc="train"):
        x = x.to(device)   # (B,1,28,28)
        y = y.to(device)   # (B,)

        spikes = enc(x)    # (T,B,1,28,28)
        spikes = spikes.to(device)

        logits = model(spikes)     # (B,10)
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
def eval_epoch(model, enc, dl, device, T):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in tqdm(dl, desc="eval"):
        x = x.to(device)
        y = y.to(device)
        spikes = enc(x).to(device)
        logits = model(spikes)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        total_acc  += accuracy(logits, y) * x.size(0)
        n += x.size(0)

    return total_loss / n, total_acc / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--T", type=int, default=25, help="passos de tempo do Poisson encoder")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true", help="força CPU (default usa cuda se disponível)")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)
    enc = PoissonEncoder(T=args.T).to(device)
    model = SNNConvNet().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, enc, train_dl, opt, device, args.T)
        te_loss, te_acc = eval_epoch(model, enc, test_dl, device, args.T)
        dt = time.time() - t0
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={te_loss:.4f} acc={te_acc:.4f} | {dt:.1f}s")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "snn_mnist_lif_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
