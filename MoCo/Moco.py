import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- PARÂMETROS DE CONFIGURAÇÃO ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
LEARNING_RATE_MOCO = 1e-3
LEARNING_RATE_CLASSIFIER = 1e-3
EPOCHS_MOCO = 10
EPOCHS_CLASSIFIER = 10
MOCO_DIM = 128
MOCO_K = 4096
MOCO_M = 0.999
TEMPERATURE = 0.07

# ===================================================================
# 1. DEFINIÇÃO DOS MODELOS (ENCODER, MOCO, CLASSIFICADOR)
# ===================================================================
class Encoder(nn.Module):
    def __init__(self, feature_dim=MOCO_DIM):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    def forward(self, x):
        x = self.convnet(x)
        x = self.fc(x)
        return F.normalize(x, dim=1)

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=MOCO_DIM, K=MOCO_K, m=MOCO_M, T=TEMPERATURE):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(feature_dim=dim)
        self.encoder_k = base_encoder(feature_dim=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)
        self._dequeue_and_enqueue(k)
        return logits, labels

class LinearClassifier(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder
        self.classifier_head = nn.Linear(MOCO_DIM, 10)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier_head(features)

# ===================================================================
# 2. FUNÇÕES DE DADOS E TREINO
# ===================================================================
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def get_loaders_moco(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=TwoCropsTransform(train_transform))
    # [CORREÇÃO APLICADA AQUI] drop_last=True evita o erro de tamanho no último lote
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader

def get_loaders_classifier(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    print(f"Usando dispositivo: {DEVICE}")

    # FASE 1: MOCO
    print("\n--- FASE 1: INICIANDO PRÉ-TREINAMENTO COM MOCO ---")
    moco_loader = get_loaders_moco(BATCH_SIZE)
    moco_model = MoCo(Encoder).to(DEVICE)
    criterion_moco = nn.CrossEntropyLoss()
    optimizer_moco = optim.Adam(moco_model.parameters(), lr=LEARNING_RATE_MOCO)

    for epoch in range(EPOCHS_MOCO):
        loop = tqdm(moco_loader, leave=True)
        loop.set_description(f"Epoch [MoCo] {epoch+1}/{EPOCHS_MOCO}")
        for images, _ in loop:
            im_q, im_k = images[0].to(DEVICE), images[1].to(DEVICE)
            output, target = moco_model(im_q, im_k)
            loss = criterion_moco(output, target)
            optimizer_moco.zero_grad()
            loss.backward()
            optimizer_moco.step()
            loop.set_postfix(loss=loss.item())
    print("--- FASE 1: PRÉ-TREINAMENTO MOCO CONCLUÍDO ---")

    # FASE 2: CLASSIFICAÇÃO
    print("\n--- FASE 2: INICIANDO TREINO DO CLASSIFICADOR (TRANSFER LEARNING) ---")
    train_loader, test_loader = get_loaders_classifier(BATCH_SIZE)
    trained_encoder = moco_model.encoder_q
    classifier_model = LinearClassifier(pretrained_encoder=trained_encoder).to(DEVICE)
    
    print("Congelando pesos do encoder pré-treinado...")
    for param in classifier_model.encoder.parameters():
        param.requires_grad = False
        
    criterion_classifier = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier_model.classifier_head.parameters(), lr=LEARNING_RATE_CLASSIFIER)

    for epoch in range(EPOCHS_CLASSIFIER):
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [Classifier] {epoch+1}/{EPOCHS_CLASSIFIER}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = classifier_model(imgs)
            loss = criterion_classifier(outputs, labels)
            optimizer_classifier.zero_grad()
            loss.backward()
            optimizer_classifier.step()
            loop.set_postfix(loss=loss.item())
    print("--- FASE 2: TREINO DO CLASSIFICADOR CONCLUÍDO ---")
    
    # Avaliação
    print("\n--- AVALIANDO O CLASSIFICADOR FINAL NO DATASET DE TESTE ---")
    classifier_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = classifier_model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'\nAcurácia do classificador (treinado com features do MoCo): {accuracy:.2f} %')

if __name__ == '__main__':
    main()