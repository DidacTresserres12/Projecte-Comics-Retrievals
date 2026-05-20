import wandb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from datasets import load_dataset
from pytorch_metric_learning import losses
import numpy as np
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.models import Baseline
from dataloaders.dataset import ComicsPAPDataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    return loader


def make(config, device="cuda", use_subset=True):
    """
    Construeix model, dataloaders, optimitzador i funció de pèrdua.

    Millores respecte a la versió original:
      1. SentenceBERT (all-MiniLM-L6-v2) en lloc de BERT base:
         embeddings de frases entrenats per similitud cosinus + mean pooling
      2. TripletMarginLoss: mateixa loss que abans, provada i estable
      3. Batch sizes fins a 256 (512 donava OOM a la GPU)
    """

    raw_dataset = load_dataset("/home/datasets/COMIC-PAP")

    embedding_dim = getattr(config, "embedding_dim", 512)
    freeze_sbert  = getattr(config, "freeze_sbert",  True)
    freeze_resnet = getattr(config, "freeze_resnet", True)

    NUM_WORKERS = 4

    model = Baseline(
        embedding_dim=embedding_dim,
        freeze_sbert=freeze_sbert,
        freeze_resnet=freeze_resnet,
    ).to(device)

    tokenizer = model.sbert_tokenizer

    full_train_data = ComicsPAPDataset(raw_dataset['train'],      tokenizer)
    test_data       = ComicsPAPDataset(raw_dataset['validation'], tokenizer)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )

    # ── Funció de pèrdua: TripletMarginLoss amb in-batch negatives ────────────
    #
    # TripletLoss: donat (anchor, positive, negative):
    #   L = max(0, d(a, p) - d(a, n) + margin)
    #
    # Com funciona el mining en el nostre cas?
    #   A train_batch concatenem els embeddings de text i imatge amb labels
    #   iguals per a cada parella (text_i, img_i) → label i.
    #   pytorch_metric_learning mina automàticament tots els triplets vàlids
    #   del batch: per cada anchor té N-1 negatius disponibles (tots els
    #   exemples del batch amb label diferent).
    #   Amb batch gran (128, 256) → més triplets → millor senyal de gradient.
    #
    # Per què TripletLoss i no QuadrupletLoss?
    #   Vam provar QuadrupletLoss però no va millorar els resultats i
    #   afegeix complexitat d'hiperparàmetres (margin1, margin2).
    #   TripletLoss és més estable i suficient per a aquest dataset.
    # ─────────────────────────────────────────────────────────────────────────
    criterion = losses.TripletMarginLoss(
        margin=getattr(config, "margin", 0.2)
    )

    return model, train_loader, test_loader, criterion, optimizer
