# gan_classifier.py
# ------------------------------------------------------------
# Usando o Discriminador de uma GAN como um classificador.
# Fase 1: Treina a GAN de forma não supervisionada para aprender features.
# Fase 2: Usa o discriminador treinado, congela suas camadas, e treina
#         uma nova cabeça de classificação de forma supervisionada.
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
# 1) ARQUITETURAS: GERADOR E DISCRIMINADOR
# ============================================================

class Generator(nn.Module):
    """ O Falsificador: transforma um vetor de ruído em uma imagem. """
    def __init__(self, latent_dim=100, img_shape=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_shape),
            nn.Tanh() # Tanh escala a saída para [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """ O Detetive: classifica uma imagem como real (1) ou falsa (0). """
    def __init__(self, img_shape=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Sigmoid para output de probabilidade [0, 1]
        )

    def forward(self, img):
        return self.net(img)

# ============================================================
# 2) O NOVO MODELO: CLASSIFICADOR BASEADO NO DISCRIMINADOR
# ============================================================
class GANClassifier(nn.Module):
    def __init__(self, trained_discriminator, num_classes=10):
        super().__init__()
        # Pega as camadas de features do discriminador (todas menos as duas últimas: Linear(256,1) e Sigmoid)
        self.features = nn.Sequential(*list(trained_discriminator.net.children())[:-2])
        # Adiciona a nova cabeça de classificação
        self.classifier = nn.Linear(256, num_classes) # A última camada de features tinha 256 neurônios

    def forward(self, img):
        # Extrai features com as camadas congeladas
        feat = self.features(img)
        # Classifica com a nova cabeça
        return self.classifier(feat)

# ============================================================
# 3) LÓGICA DE TREINO E AVALIAÇÃO
# ============================================================
def get_loaders(batch_size=64):
    # Normalizamos as imagens para o range [-1, 1] para combinar com a saída Tanh do gerador
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
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
        x = x.view(x.size(0), -1).to(device)
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
def eval_classifier_epoch(model, dl, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in tqdm(dl, desc="Classifier Eval"):
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(dim=-1) == y).float().sum().item()
        n += x.size(0)
    return total_loss / n, total_acc / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gan_epochs", type=int, default=20, help="Épocas para treinar a GAN (Fase 1)")
    p.add_argument("--classifier_epochs", type=int, default=5, help="Épocas para treinar o classificador (Fase 2)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")
    train_dl, test_dl = get_loaders(args.batch_size)
    
    # --- FASE 1: TREINO NÃO SUPERVISIONADO DA GAN ---
    print("\n--- FASE 1: TREINANDO A GAN ---")
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    for epoch in range(1, args.gan_epochs + 1):
        for real_imgs, _ in tqdm(train_dl, desc=f"GAN Epoch {epoch}"):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Treina Discriminador
            d_opt.zero_grad()
            d_loss = adversarial_loss(discriminator(real_imgs), real_labels) + \
                     adversarial_loss(discriminator(generator(torch.randn(batch_size, args.latent_dim, device=device)).detach()), fake_labels)
            d_loss.backward()
            d_opt.step()
            
            # Treina Gerador
            g_opt.zero_grad()
            g_loss = adversarial_loss(discriminator(generator(torch.randn(batch_size, args.latent_dim, device=device))), real_labels)
            g_loss.backward()
            g_opt.step()
        print(f"[GAN Epoch {epoch}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # --- FASE 2: TREINO SUPERVISIONADO DO CLASSIFICADOR ---
    print("\n--- FASE 2: TREINANDO O CLASSIFICADOR ---")
    
    # Cria o modelo classificador usando o discriminador treinado
    classifier_model = GANClassifier(discriminator).to(device)
    
    # CONGELA as camadas de features
    for param in classifier_model.features.parameters():
        param.requires_grad = False
    
    # Otimizador treinará APENAS os parâmetros da nova cabeça de classificação
    classifier_opt = torch.optim.Adam(classifier_model.classifier.parameters(), lr=1e-3)
    
    best_acc = 0.0
    for epoch in range(1, args.classifier_epochs + 1):
        tr_loss, tr_acc = train_classifier_epoch(classifier_model, train_dl, classifier_opt, device)
        te_loss, te_acc = eval_classifier_epoch(classifier_model, test_dl, device)
        print(f"[Classifier Epoch {epoch}] Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Test Loss={te_loss:.4f} Acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(classifier_model.state_dict(), "gan_classifier_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()