import os
import sys
import random
import numpy as np
import torch
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils.utils import make
from train import train

# ── Reproducibilitat ──────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Millors hiperparàmetres: firm-sweep-24 ────────────────────────────────────
# Run: firm-sweep-24  (WandB project: Dataset_weight decay)
# Mètriques assolides en el sweep (20 epochs, 40% subset):
#   val/best_R@1  = 0.0458
#   val/best_R@10 = 0.2328
#   val/best_MRR  = 0.1002
#
# En aquesta execució final s'utilitzen 150 epochs i el 40% del dataset
# (fracció establerta a utils.py) per aprofitar al màxim el temps de còmput
# disponible i veure si el model continua millorant.
# ─────────────────────────────────────────────────────────────────────────────
config_dict = {
    "learning_rate":  1e-4,
    "batch_size":     64,
    "embedding_dim":  512,
    "margin":         0.4,
    "clip_threshold": 0.29,
    "freeze_sbert":   True,   # SBERT congelat (recomanació prof.)
    "freeze_resnet":  True,   # ResNet50 layer4 es descongela a train.py
    "epochs":         150,
    "dataset":        "ComicsPAP",
    "architecture":   "ResNet50-SBERTcongelat-TripletLoss",
    "weight_decay":   0.1,    # AdamW weight decay (utils.py)
    "subset_fraction": 0.40,  # 40% del dataset d'entrenament
}

SAVE_PATH = os.path.join(script_dir, "trained_models", "model_final_150ep.pth")


if __name__ == "__main__":
    wandb.login()

    with wandb.init(
        project="Comics-Retrieval-Final",
        name="firm-sweep-24_150epochs",
        config=config_dict,
        tags=["final", "best-hparams", "150ep"],
    ) as run:
        config = wandb.config

        print(f"\n{'='*60}")
        print(f"  RUN FINAL — millors hiperparàmetres (firm-sweep-24)")
        print(f"{'='*60}")
        print(f"  learning_rate  : {config.learning_rate}")
        print(f"  batch_size     : {config.batch_size}")
        print(f"  embedding_dim  : {config.embedding_dim}")
        print(f"  margin         : {config.margin}")
        print(f"  clip_threshold : {config.clip_threshold}")
        print(f"  epochs         : {config.epochs}")
        print(f"  freeze_sbert   : {config.freeze_sbert}")
        print(f"  weight_decay   : {config.weight_decay}")
        print(f"{'='*60}\n")

        model, train_loader, val_loader, criterion, optimizer = make(
            config, device=device
        )

        best_r1 = train(
            model=model,
            loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        # Guardem el model final (a més del millor epoch que ja guarda train.py)
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save({
            "config":   dict(config),
            "model":    model.state_dict(),
            "best_r1":  best_r1,
        }, SAVE_PATH)
        print(f"\n✅ Model final guardat a: {SAVE_PATH}")
        print(f"   Millor R@1 assolit: {best_r1:.4f}")
