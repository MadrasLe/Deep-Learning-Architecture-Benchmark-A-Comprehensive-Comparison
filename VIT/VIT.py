# vit_mnist.py
# ------------------------------------------------------------
# Vision Transformer (ViT) em PyTorch para classificação no MNIST
# - Substitui a arquitetura CNN
# - Arquitetura ViT adaptada para imagens pequenas (28x28)
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
# 1) ARQUITETURA VISION TRANSFORMER (ViT)
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Converte uma imagem em uma sequência de embeddings de patches.
    (B, C, H, W) -> (B, N, D)
    onde N = (H*W)/(P*P) é o número de patches e D é a dimensão do embedding.
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Uma convolução é a forma mais eficiente de criar patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x

class VisionTransformer(nn.Module):
    """
    Arquitetura Vision Transformer simplificada para MNIST.
    """
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=128,      # Dimensão da projeção linear dos patches
        depth=4,            # Número de blocos Transformer
        num_heads=4,        # Número de cabeças de atenção
        mlp_dim=256,        # Dimensão da camada oculta no MLP do Transformer
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Token [CLS] para classificação
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Embeddings de Posição (aprendíveis)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # Stacking dos blocos Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True, # Importante: nossa entrada é (B, Seq, Dim)
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # Cabeça de classificação
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        # 1. Patching e Embedding
        x = self.patch_embed(x)  # (B, N, D)

        # 2. Adicionar token [CLS]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        # 3. Adicionar embeddings de posição
        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        # 4. Passar pelo Transformer Encoder
        x = self.transformer_encoder(x)  # (B, N+1, D)

        # 5. Extrair o token [CLS] e classificar
        cls_output = x[:, 0]  # Pega a saída do primeiro token
        logits = self.mlp_head(cls_output) # (B, num_classes)
        return logits


# ============================================================
# 2) TREINO / AVALIAÇÃO (Idênticos ao da CNN)
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
        x = x.to(device)
        y = y.to(device)

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
    p.add_argument("--epochs", type=int, default=5) # ViTs podem precisar de mais épocas
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true", help="força CPU (default usa cuda se disponível)")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    train_dl, test_dl = get_loaders(args.batch_size)

    # Instancia o modelo Vision Transformer
    model = VisionTransformer(
        patch_size=4,  # 28/4 = 7 patches por lado -> 49 patches no total
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256
    ).to(device)
    print(model)

    # Contar parâmetros para curiosidade
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
            torch.save(model.state_dict(), "vit_mnist_best.pt")
            print(f"💾 Checkpoint salvo (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()