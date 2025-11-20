import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# --- PARÂMETROS DE CONFIGURAÇÃO ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512 # SimSiam se beneficia de batch sizes maiores
LEARNING_RATE_SSL = 1e-3
EPOCHS_SSL = 5 # Vamos dar um pouco mais de épocas para ele convergir bem
PROJECTION_DIM = 2048 # Dimensão da projeção (antes da predição)
PREDICTION_DIM = 512  # Dimensão da cabeça de predição
FEATURE_DIM = 2048    # Dimensão final do embedding

# ===================================================================
# 1. DEFINIÇÃO DOS MODELOS (ENCODER, SIMSIAM)
# ===================================================================

# O Encoder pode ser o mesmo
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2) # Saída: [batch, 128, 3, 3]
        )
        self.fc = nn.Flatten()
        
    def forward(self, x):
        return self.fc(self.convnet(x))

class SimSiam(nn.Module):
    def __init__(self, base_encoder):
        super(SimSiam, self).__init__()
        self.encoder = base_encoder()
        
        # Cabeça de Projeção (Projector)
        self.projector = nn.Sequential(
            nn.Linear(128 * 3 * 3, PROJECTION_DIM),
            nn.BatchNorm1d(PROJECTION_DIM),
            nn.ReLU(),
            nn.Linear(PROJECTION_DIM, FEATURE_DIM)
        )
        
        # Cabeça de Predição (Predictor)
        self.predictor = nn.Sequential(
            nn.Linear(FEATURE_DIM, PREDICTION_DIM),
            nn.BatchNorm1d(PREDICTION_DIM),
            nn.ReLU(),
            nn.Linear(PREDICTION_DIM, FEATURE_DIM)
        )
    
    def forward(self, x1, x2):
        # Passa as duas visões pelo encoder
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        
        # Passa pelo projetor
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        
        # Passa um dos ramos pelo preditor
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Aplica o stop-gradient e calcula a loss
        loss = -(F.cosine_similarity(p1, z2.detach()).mean() + F.cosine_similarity(p2, z1.detach()).mean()) * 0.5
        
        return loss

# ===================================================================
# 2. FUNÇÕES DE DADOS E TREINO
# ===================================================================
class TwoCropsTransform:
    def __init__(self, base_transform): self.base_transform = base_transform
    def __call__(self, x): return [self.base_transform(x), self.base_transform(x)]

def get_ssl_loader(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=TwoCropsTransform(train_transform))
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

def get_classifier_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def extract_embeddings(encoder, data_loader, device):
    encoder.eval()
    embeddings, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Extraindo Embeddings"):
            features = encoder(imgs.to(device))
            embeddings.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels_list)

def main():
    print(f"Usando dispositivo: {DEVICE}")

    # ====================================================================================
    # FASE 1: PRÉ-TREINAMENTO AUTO-SUPERVISIONADO (SIMSIAM)
    # ====================================================================================
    print("\n--- FASE 1: INICIANDO PRÉ-TREINAMENTO COM SIMSIAM ---")
    ssl_loader = get_ssl_loader(BATCH_SIZE)
    siam_model = SimSiam(Encoder).to(DEVICE)
    optimizer = optim.Adam(siam_model.parameters(), lr=LEARNING_RATE_SSL)

    for epoch in range(EPOCHS_SSL):
        loop = tqdm(ssl_loader, leave=True)
        loop.set_description(f"Epoch [SimSiam] {epoch+1}/{EPOCHS_SSL}")
        for images, _ in loop:
            x1, x2 = images[0].to(DEVICE), images[1].to(DEVICE)
            loss = siam_model(x1, x2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loop.set_postfix(loss=loss.item())
    print("--- FASE 1: PRÉ-TREINAMENTO SIMSIAM CONCLUÍDO ---")

    # ====================================================================================
    # FASE 2: CLASSIFICAÇÃO COM GBM
    # ====================================================================================
    print("\n--- FASE 2: PREPARANDO DADOS PARA O CLASSIFICADOR GBM ---")
    train_loader, test_loader = get_classifier_loaders(BATCH_SIZE)
    
    # Extrai o encoder treinado, nosso gerador de embeddings
    trained_encoder = siam_model.encoder
    
    X_train, y_train = extract_embeddings(trained_encoder, train_loader, DEVICE)
    X_test, y_test = extract_embeddings(trained_encoder, test_loader, DEVICE)
    print(f"Embeddings de treino gerados. Shape: {X_train.shape}")
    print(f"Embeddings de teste gerados. Shape: {X_test.shape}")

    print("\n--- FASE 2: TREINANDO O CLASSIFICADOR GBM (LIGHTGBM) ---")
    gbm_classifier = lgb.LGBMClassifier(objective='multiclass', num_class=10, random_state=42)
    gbm_classifier.fit(X_train, y_train)
    print("--- FASE 2: TREINO DO GBM CONCLUÍDO ---")
    
    print("\n--- AVALIANDO O CLASSIFICADOR GBM NO DATASET DE TESTE ---")
    y_pred = gbm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'\nAcurácia do classificador GBM (com features do SimSiam): {accuracy:.2f} %')

if __name__ == '__main__':
    main()