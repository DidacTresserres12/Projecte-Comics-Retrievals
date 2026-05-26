import wandb
import torch
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

CLIP_SCORES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clip_scores")


def make(config, device="cuda", use_subset=True):
    """
    Construeix model, dataloaders, optimitzador i funció de pèrdua.

    Lògica de subset:
      - use_subset=False → 100% de les dades sempre
      - use_subset=True  + clip_threshold == 0.0 → 20% de les dades
      - use_subset=True  + clip_threshold >  0.0 → 20% de les dades
          (el filtre CLIP ja elimina mostres senceres, el subset s'aplica
           sobre les que queden)

    El filtratge de CLIP score elimina mostres senceres (aquelles on el
    primer paràgraf de la caption té score < threshold). Això garanteix
    que les parelles text-imatge que resten estan ben alineades.
    """
    raw_dataset   = load_dataset("/home/datasets/COMIC-PAP")
    embedding_dim = getattr(config, "embedding_dim",  512)
    freeze_sbert  = getattr(config, "freeze_sbert",   True)
    freeze_resnet = getattr(config, "freeze_resnet",  True)
    clip_thr      = getattr(config, "clip_threshold", 0.0)

    NUM_WORKERS = 4

    model     = Baseline(embedding_dim=embedding_dim,
                         freeze_sbert=freeze_sbert,
                         freeze_resnet=freeze_resnet).to(device)
    tokenizer = model.sbert_tokenizer

    # Rutes dels CLIP scores precomputats (un float per mostra, pel primer paràgraf)
    clip_train_path = os.path.join(CLIP_SCORES_DIR, "clip_scores_train.json")
    clip_val_path   = os.path.join(CLIP_SCORES_DIR, "clip_scores_validation.json")

    full_train_data = ComicsPAPDataset(
        raw_dataset['train'], tokenizer,
        clip_scores_path=clip_train_path if clip_thr > 0 else None,
        clip_threshold=clip_thr
    )
    # Validació: SENSE clip_threshold → avaluem sobre el dataset complet.
    # El clip filtra pars de baixa qualitat per al training, però en val
    # volem mesurar el rendiment real sobre TOTES les mostres.
    val_data = ComicsPAPDataset(
        raw_dataset['validation'], tokenizer,
        clip_scores_path=None,
        clip_threshold=0.0
    )

    # ── Selecció del subset d'entrenament ────────────────────────────────────
    if use_subset:
        num_train = len(full_train_data)   # ja filtrat per CLIP si clip_thr > 0
        rng = np.random.default_rng(42)

        fraction   = 0.40
        n_subset   = int(num_train * fraction)
        subset_indices = rng.choice(num_train, n_subset, replace=False)
        train_data     = Subset(full_train_data, subset_indices)
        print(f"  [Subset] {int(fraction*100)}% del dataset filtrat: "
              f"{len(train_data)} mostres de {num_train} "
              f"(clip_threshold={clip_thr:.2f})")
    else:
        train_data = full_train_data
        print(f"  [Subset] Dataset complet: {len(train_data)} mostres "
              f"(clip_threshold={clip_thr:.2f})")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=config.batch_size,
        shuffle=True, pin_memory=True, num_workers=NUM_WORKERS,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=config.batch_size,
        shuffle=False, pin_memory=True, num_workers=NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.1
    )

    criterion = losses.TripletMarginLoss(
        margin=getattr(config, "margin", 0.2)
    )

    return model, train_loader, val_loader, criterion, optimizer