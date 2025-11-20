# vae_classifier_beta.py
# ------------------------------------------------------------
# Beta-Variational Autoencoder (β-VAE) usado para classificação.
# O parâmetro beta controla o peso da perda KLD. Um beta < 1
# foca mais na reconstrução, gerando features mais úteis para
# a classificação.
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
# 1) ARQUITETURA VAE
# ============================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: Imagem -> Média (mu) e Log da Variância (logvar)
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2) # Saída para mu e logvar
        )
        
        # Decoder: Vetor Latente (z) -> Imagem
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid() # Sigmoid para output de pixel [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """ O "truque" de reparametrização para permitir o backpropagation. """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Codifica a entrada e separa em mu e logvar
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        
        # Amostra um ponto z do espaço latente
        z = self.reparameterize(mu, logvar)
        
        # Decodifica z para reconstruir a imagem
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# Função de perda do VAE (Reconstrução + KL Divergence) com o peso beta
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Usamos Binary Cross Entropy como perda de reconstrução
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL Divergence: a penalidade para organizar o espaço latente
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # A perda KLD agora é multiplicada por beta
    return BCE + beta * KLD

# ============================================================
# 2) O CLASSIFICADOR BASEADO NO ENCODER
# ============================================================
class VAEClassifier(nn.Module):
    def __init__(self, trained_vae, num_classes=10):
        super().__init__()
        self.encoder = trained_vae.encoder
        self.classifier = nn.Linear(trained_vae.latent_dim, num_classes)

    def forward(self, x):
        # Usa o encoder para extrair features (mu e logvar)
        h = self.encoder(x)
        mu, _ = h.chunk(2, dim=-1) # Só precisamos de mu para classificar
        return self.classifier(mu)

# ============================================================
# 3) LÓGICA DE TREINO E AVALIAÇÃO
# ============================================================
def get_loaders(batch_size=128):
    tfm = transforms.Compose([transforms.ToTensor()])
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
        x, y = x.view(x.size(0), -1).to(device), y.to(device)
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
def eval_classifier_epoch(model, dl, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="Classifier Eval"):
        x, y = x.view(x.size(0), -1).to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).float().sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_epochs", type=int, default=10)
    p.add_argument("--classifier_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=20)
    # Adicionado o argumento beta
    p.add_argument("--beta", type=float, default=1.0, help="Peso da perda KLD (beta < 1 foca em reconstrução)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    output_dir = "vae_outputs"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")
    train_dl, test_dl = get_loaders(args.batch_size)

    # --- FASE 1: TREINO NÃO SUPERVISIONADO DO VAE ---
    print("\n--- FASE 1: TREINANDO O VAE ---")
    vae_model = VAE(args.latent_dim).to(device)
    vae_opt = torch.optim.Adam(vae_model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.vae_epochs + 1):
        vae_model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(train_dl, desc=f"VAE Epoch {epoch}")):
            data = data.view(-1, 784).to(device)
            vae_opt.zero_grad()
            recon_batch, mu, logvar = vae_model(data)
            # Passa o beta para a função de perda
            loss = loss_function(recon_batch, data, mu, logvar, args.beta)
            loss.backward()
            train_loss += loss.item()
            vae_opt.step()
        print(f"Epoch {epoch} Avg. VAE Loss: {train_loss / len(train_dl.dataset):.4f} (beta={args.beta})")

        with torch.no_grad():
            sample = torch.randn(64, args.latent_dim).to(device)
            generated = vae_model.decoder(sample).cpu()
            save_image(generated.view(64, 1, 28, 28), os.path.join(output_dir, f'generated_epoch_{epoch}_beta{args.beta}.png'))

    # --- FASE 2: TREINO SUPERVISIONADO DO CLASSIFICADOR ---
    print("\n--- FASE 2: TREINANDO O CLASSIFICADOR ---")
    classifier_model = VAEClassifier(vae_model).to(device)
    
    # Congela os pesos do encoder
    for param in classifier_model.encoder.parameters():
        param.requires_grad = False
        
    classifier_opt = torch.optim.Adam(classifier_model.classifier.parameters(), lr=args.lr)
    
    best_acc = 0.0
    for epoch in range(1, args.classifier_epochs + 1):
        tr_loss, tr_acc = train_classifier_epoch(classifier_model, train_dl, classifier_opt, device)
        te_loss, te_acc = eval_classifier_epoch(classifier_model, test_dl, device)
        print(f"[Classifier Epoch {epoch}] Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Test Loss={te_loss:.4f} Acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(classifier_model.state_dict(), "vae_classifier_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()