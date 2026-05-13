import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from datasets import load_dataset
import numpy as np
import os
import sys
from pytorch_metric_learning import losses

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.models import *
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


def make(config, device="cuda", use_subset=True):
    # Carregar Dataset des de la ruta local de la VM
    raw_dataset = load_dataset("/home/datasets/COMIC-PAP")

    # ── Llegim els paràmetres del sweep ──
    embedding_dim  = getattr(config, "embedding_dim",  512)
    freeze_bert    = getattr(config, "freeze_bert",    True)
    freeze_resnet  = getattr(config, "freeze_resnet",  True)

    # num_workers: paràmetre de sistema, NO del sweep.
    # 4 és segur per a VMs amb 8 CPU; baixa a 2 si hi ha errors de memòria.
    NUM_WORKERS = 4

    model = Baseline(
        embedding_dim=embedding_dim,
        freeze_bert=freeze_bert,
        freeze_resnet=freeze_resnet,
    ).to(device)

    tokenizer = model.bert_tokenizer

    # Dataset complet
    full_train_data = ComicsPAPDataset(raw_dataset['train'], tokenizer)
    test_data = ComicsPAPDataset(raw_dataset['validation'],   tokenizer)

    # Reducció al 20% del train per al sweep (reproducible amb seed fixe)
    if use_subset:
        num_train = len(full_train_data)
        rng = np.random.default_rng(42)
        subset_indices = rng.choice(num_train, int(num_train * 0.2), replace=False)
        train_data = Subset(full_train_data, subset_indices)
        print(f"Usant SUBSET del 20%: {len(train_data)} mostres de {num_train}")
    else:
        train_data = full_train_data
        print(f"Usant DATASET COMPLET: {len(train_data)} mostres")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Margin llegit de config (hiperparàmetre del sweep)
    criterion = losses.TripletMarginLoss(margin=getattr(config, "margin", 0.2))

    return model, train_loader, test_loader, criterion, optimizer
