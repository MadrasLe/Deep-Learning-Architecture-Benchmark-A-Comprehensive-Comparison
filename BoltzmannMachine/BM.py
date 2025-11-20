# rbm_mnist.py
# ------------------------------------------------------------
# Máquina de Boltzmann Restrita (RBM) em PyTorch para o MNIST
# - Modelo não supervisionado, aprende features dos dados.
# - Treinado com Divergência Contrastiva (CD-1).
# - Avaliação por erro de reconstrução e visualização de features.
# ------------------------------------------------------------
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x

# ============================================================
# 1) ARQUITETURA RBM
# ============================================================
class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=500):
        super().__init__()
        # Inicializa os parâmetros: pesos (W) e biases para as camadas visível e oculta
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def prob_h_given_v(self, v):
        """ Calcula a probabilidade de ativação dos neurônios ocultos dado o estado dos visíveis. """
        return torch.sigmoid(F.linear(v, self.W, self.h_bias))

    def prob_v_given_h(self, h):
        """ Calcula a probabilidade de ativação dos neurônios visíveis dado o estado dos ocultos (reconstrução). """
        return torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))

    def sample_h_given_v(self, v):
        """ Amostra o estado dos neurônios ocultos (0 ou 1). """
        return torch.bernoulli(self.prob_h_given_v(v))

    def sample_v_given_h(self, h):
        """ Amostra o estado dos neurônios visíveis (0 ou 1). """
        return torch.bernoulli(self.prob_v_given_h(h))

# ============================================================
# 2) TREINO / AVALIAÇÃO
# ============================================================
def get_loaders(batch_size=64):
    # Para RBMs, queremos os pixels brutos em [0,1], sem normalização de média/desvio padrão
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl

def train_epoch(model, dl, device, lr):
    model.train()
    total_recon_error = 0.0

    for x, _ in tqdm(dl, desc="train"):
        x = x.view(-1, 784).to(device)
        # RBMs trabalham com dados binários, então tratamos os pixels como probabilidades e amostramos
        v0 = torch.bernoulli(x)

        # --- Início da Divergência Contrastiva (CD-1) ---

        # Fase Positiva: Ativações ocultas a partir dos dados reais
        # Usamos as probabilidades ph0 para a atualização, pois é mais estável
        ph0 = model.prob_h_given_v(v0)
        h0 = torch.bernoulli(ph0) # Amostra para a reconstrução

        # Fase Negativa: Reconstrução a partir das ativações ocultas
        v1 = model.sample_v_given_h(h0)
        ph1 = model.prob_h_given_v(v1)

        # --- Fim da CD-1 ---

        # Cálculo dos gradientes
        positive_grad = torch.matmul(v0.t(), ph0)
        negative_grad = torch.matmul(v1.t(), ph1)

        # Atualização manual dos pesos e biases (sem otimizador!)
        batch_size = x.size(0)
        model.W.data += lr * (positive_grad - negative_grad).t() / batch_size
        model.v_bias.data += lr * torch.mean(v0 - v1, dim=0)
        model.h_bias.data += lr * torch.mean(ph0 - ph1, dim=0)

        total_recon_error += F.mse_loss(v1, v0, reduction='sum').item()

    return total_recon_error / len(dl.dataset)

@torch.no_grad()
def eval_epoch(model, dl, device, epoch, output_dir):
    model.eval()
    total_recon_error = 0.0

    # Pega um batch fixo para visualizar as reconstruções
    fixed_x, _ = next(iter(dl))
    fixed_x = fixed_x.to(device)
    fixed_v0 = torch.bernoulli(fixed_x.view(-1, 784))

    # Reconstrução
    h = model.sample_h_given_v(fixed_v0)
    v1 = model.sample_v_given_h(h)
    
    # Salva a comparação entre original e reconstruído
    comparison = torch.cat([fixed_v0.view(-1, 1, 28, 28), v1.view(-1, 1, 28, 28)])
    save_image(comparison.cpu(), os.path.join(output_dir, f"reconstruction_epoch_{epoch}.png"), nrow=fixed_x.size(0))

    for x, _ in tqdm(dl, desc="eval"):
        x = x.view(-1, 784).to(device)
        v0 = torch.bernoulli(x)
        h = model.sample_h_given_v(v0)
        v1 = model.sample_v_given_h(h)
        total_recon_error += F.mse_loss(v1, v0, reduction='sum').item()

    return total_recon_error / len(dl.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    output_dir = "rbm_outputs"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)
    model = RBM(n_visible=784, n_hidden=512).to(device)
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número total de parâmetros: {num_params:,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_err = train_epoch(model, train_dl, device, args.lr)
        te_err = eval_epoch(model, test_dl, device, epoch, output_dir)
        dt = time.time() - t0
        print(f"[Epoch {epoch}] Train Recon Error={tr_err:.4f} | Test Recon Error={te_err:.4f} | {dt:.1f}s")
        
        # Salva as features (pesos) que o modelo aprendeu
        weights = model.W.cpu().data
        save_image(weights.view(-1, 1, 28, 28), os.path.join(output_dir, f"weights_epoch_{epoch}.png"), nrow=32)

if __name__ == "__main__":
    main()