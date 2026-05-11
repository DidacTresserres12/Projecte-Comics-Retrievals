import wandb
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import os
import sys
from pytorch_metric_learning import losses

# Calculamos la ruta de la carpeta raíz del proyecto
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.models import *
from datasets import load_dataset
from dataloaders.dataset import ComicsPAPDataset


def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Carregar Dataset
    raw_dataset = load_dataset("VLR-CVC/ComicsPAP", "caption_relevance")

    # ── Llegim els paràmetres del sweep (amb valors per defecte per compatibilitat) ──
    embedding_dim  = getattr(config, "embedding_dim",  512)
    freeze_bert    = getattr(config, "freeze_bert",    True)
    freeze_resnet  = getattr(config, "freeze_resnet",  True)

    # Instanciar Model amb els paràmetres del sweep
    model = Baseline(
        embedding_dim=embedding_dim,
        freeze_bert=freeze_bert,
        freeze_resnet=freeze_resnet,
    ).to(device)

    tokenizer = model.bert_tokenizer

    # Crear Datasets i Loaders
    train_data = ComicsPAPDataset(raw_dataset['train'], tokenizer)
    test_data  = ComicsPAPDataset(raw_dataset['val'],   tokenizer)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=config.batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=config.batch_size,
        shuffle=False, pin_memory=True)

    # Optimitzador
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

    return model, train_loader, test_loader, criterion, optimizer