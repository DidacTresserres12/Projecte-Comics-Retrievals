import wandb
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset
from datasets import load_dataset
from pytorch_metric_learning import losses
import numpy as np
import os
import sys
import json
import io
from PIL import Image

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from models.models import Baseline
from dataloaders.dataset import ComicsPAPDataset

CLIP_SCORES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_scores")


def make_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=2
    )


def make(config, device="cuda", use_subset=True):
    """
    Construeix model, dataloaders, optimitzador i funció de pèrdua.

    Millores implementades:
      1. SentenceBERT (all-MiniLM-L6-v2) + mean pooling en lloc de BERT+[CLS]
      2. TripletMarginLoss amb in-batch negatives
      3. Filtratge per CLIP score (clip_threshold): elimina mostres on el text
         i la imatge correcta estan poc alineats visualment. Threshold=0.0
         desactiva el filtre (usa totes les mostres).
    """
    raw_dataset   = load_dataset("/home/datasets/COMIC-PAP")
    embedding_dim = getattr(config, "embedding_dim",   512)
    freeze_sbert  = getattr(config, "freeze_sbert",    True)
    freeze_resnet = getattr(config, "freeze_resnet",   True)
    clip_thr      = getattr(config, "clip_threshold",  0.0)

    NUM_WORKERS = 4

    model     = Baseline(embedding_dim=embedding_dim,
                         freeze_sbert=freeze_sbert,
                         freeze_resnet=freeze_resnet).to(device)
    tokenizer = model.sbert_tokenizer

    # Rutes dels CLIP scores precomputats
    clip_train_path = os.path.join(CLIP_SCORES_DIR, "clip_scores_train.json")
    clip_val_path   = os.path.join(CLIP_SCORES_DIR, "clip_scores_validation.json")

    full_train_data = ComicsPAPDataset(
        raw_dataset['train'], tokenizer,
        clip_scores_path=clip_train_path if clip_thr > 0 else None,
        clip_threshold=clip_thr
    )
    test_data = ComicsPAPDataset(
        raw_dataset['validation'], tokenizer,
        clip_scores_path=clip_val_path if clip_thr > 0 else None,
        clip_threshold=clip_thr
    )

    if use_subset and clip_thr == 0.0:
        # Subset només quan NO filtrem per CLIP (si filtrem ja reduïm el dataset)
        num_train = len(full_train_data)
        rng = np.random.default_rng(42)
        subset_indices = rng.choice(num_train, int(num_train * 0.2), replace=False)
        train_data = Subset(full_train_data, subset_indices)
        print(f"Usant SUBSET del 20%: {len(train_data)} mostres de {num_train}")
    else:
        train_data = full_train_data
        print(f"Usant dataset: {len(train_data)} mostres "
              f"(clip_threshold={clip_thr:.2f})")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=config.batch_size,
        shuffle=True, pin_memory=True, num_workers=NUM_WORKERS,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=config.batch_size,
        shuffle=False, pin_memory=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate, weight_decay=0.01)

    # ── TripletMarginLoss amb in-batch negatives ──────────────────────────────
    # Per cada anchor (text_i o img_i) amb label i, els negatius són tots els
    # embeddings amb label j≠i del mateix batch → N-1 negatius per anchor.
    # Amb batch gran (128, 256) → molts més triplets → millor senyal de gradient.
    criterion = losses.TripletMarginLoss(
        margin=getattr(config, "margin", 0.2)
    )

    return model, train_loader, test_loader, criterion, optimizer
