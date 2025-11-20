# dbn_mnist.py
# ------------------------------------------------------------
# Deep Belief Network (DBN) em PyTorch para o MNIST
# - Fase 1: Pré-treino não supervisionado, camada por camada, com RBMs.
# - Fase 2: Fine-tuning supervisionado da rede inteira para classificação.
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
# 1) RBM (A mesma classe de antes, nosso bloco de construção)
# ============================================================
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def prob_h_given_v(self, v):
        return torch.sigmoid(F.linear(v, self.W, self.h_bias))

    def sample_h_given_v(self, v):
        return torch.bernoulli(self.prob_h_given_v(v))

    def prob_v_given_h(self, h):
        return torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))

    def sample_v_given_h(self, h):
        return torch.bernoulli(self.prob_v_given_h(h))

# ============================================================
# 2) DBN (A pilha de RBMs + classificador)
# ============================================================
class DBN(nn.Module):
    def __init__(self, layer_sizes, num_classes=10):
        super().__init__()
        self.rbms = nn.ModuleList([
            RBM(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)
        ])
        self.classifier = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, x):
        for rbm in self.rbms:
            x = rbm.prob_h_given_v(x)
        return self.classifier(x)

# ============================================================
# 3) LÓGICA DE TREINO E AVALIAÇÃO
# ============================================================
def get_loaders(batch_size=64):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    )

def pretrain_rbm(rbm, dl, device, lr, epochs):
    print(f"Pre-treinando RBM: {rbm.W.shape[1]} -> {rbm.W.shape[0]}")
    for epoch in range(1, epochs + 1):
        total_recon_error = 0.0
        # ***** MUDANÇA 1: O LOOP FOI ALTERADO *****
        for batch in tqdm(dl, desc=f"Pre-train Epoch {epoch}"):
            # Pegamos apenas o primeiro elemento do batch, que são os dados
            data = batch[0]
            # ***** MUDANÇA 2: O VIEW USA O SHAPE DA RBM PARA SER GENÉRICO *****
            v0 = data.view(-1, rbm.W.shape[1]).to(device)
            
            ph0 = rbm.prob_h_given_v(v0)
            h0 = rbm.sample_h_given_v(v0)
            v1 = rbm.sample_v_given_h(h0)
            ph1 = rbm.prob_h_given_v(v1)

            positive_grad = torch.matmul(v0.t(), ph0)
            negative_grad = torch.matmul(v1.t(), ph1)
            
            batch_size = v0.size(0)
            rbm.W.data += lr * (positive_grad - negative_grad).t() / batch_size
            rbm.v_bias.data += lr * torch.mean(v0 - v1, dim=0)
            rbm.h_bias.data += lr * torch.mean(ph0 - ph1, dim=0)
            
            total_recon_error += F.mse_loss(v1, v0, reduction='sum').item()
        print(f"Recon Error: {total_recon_error / len(dl.dataset):.4f}")

def train_epoch_finetune(model, dl, opt, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="Finetune Train"):
        x = x.view(-1, 784).to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).float().sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_epoch_finetune(model, dl, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="Finetune Eval"):
        x = x.view(-1, 784).to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).float().sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--pretrain_lr", type=float, default=0.05)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)
    model = DBN(layer_sizes=[784, 512, 512]).to(device)
    print(model)

    # --- FASE 1: PRÉ-TREINO ---
    print("\n--- INICIANDO FASE DE PRÉ-TREINO ---")
    current_input_dl = train_dl
    
    for i, rbm in enumerate(model.rbms):
        pretrain_rbm(rbm, current_input_dl, device, args.pretrain_lr, args.pretrain_epochs)
        
        # Cria um novo DataLoader com as ativações da RBM atual como entrada para a próxima
        @torch.no_grad()
        def create_next_input(dl):
            new_data = []
            for batch in dl:
                data = batch[0].to(device)
                v = data.view(-1, 784)
                # Propaga através das camadas já treinadas
                for j in range(i + 1):
                    v = model.rbms[j].prob_h_given_v(v)
                new_data.append(v.cpu())
            
            new_dataset = torch.utils.data.TensorDataset(torch.cat(new_data))
            return DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Prepara os dados para a próxima RBM
        current_input_dl = create_next_input(train_dl)

    # --- FASE 2: FINE-TUNING ---
    print("\n--- INICIANDO FASE DE FINE-TUNING SUPERVISIONADO ---")
    opt = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    best_acc = 0.0
    for epoch in range(1, args.finetune_epochs + 1):
        tr_loss, tr_acc = train_epoch_finetune(model, train_dl, opt, device)
        te_loss, te_acc = eval_epoch_finetune(model, test_dl, device)
        print(f"[Finetune Epoch {epoch}] Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Test Loss={te_loss:.4f} Acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "dbn_mnist_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()