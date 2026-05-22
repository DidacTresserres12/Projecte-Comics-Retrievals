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

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Carpeta on es guardarà NOMÉS el millor model de tot el sweep
BEST_MODEL_PATH = os.path.join(script_dir, "trained_models", "model_best.pth")

sweep_config = {
    "name":   "resnet18-sbert-congelat-mrr",   # nom visible al dashboard de WandB
    "method": "bayes",
    # Monitorizamos la métrica base, WandB se encargará de gestionar el summary max
    "metric": {"name": "val/R@1", "goal": "maximize"},
    "early_terminate": {
        "type":     "hyperband",
        "min_iter": 5,
        "eta":      2,
    },
    "parameters": {
        "learning_rate": {"values": [1e-4, 2e-5, 5e-5]},
        "batch_size":    {"values": [64, 128, 256]},
        "embedding_dim": {"values": [512]},
        "margin":        {"values": [0.1, 0.2, 0.3]},
        "clip_threshold": {"values": [0.0, 0.25, 0.26, 0.27, 0.28, 0.29]},
        "freeze_sbert":  {"value": True},    # SBERT sempre congelat (prof.)
        "freeze_resnet": {"value": True},    # ResNet18 layer4 es descongela a train.py
        "epochs":        {"values": [10, 15]},
        "dataset":       {"value": "ComicsPAP"},
        "architecture":  {"value": "ResNet18-SBERTcongelat"},
    }
}

# R@1 millor global entre TOTS els runs del sweep
best_r1_global = 0.0


def model_pipeline():
    global best_r1_global

    with wandb.init() as run:
        # ============================================================
        # 📊 CONFIGURACIÓN DE LAS 6 COLUMNAS PARA EL DASHBOARD
        # ============================================================
        # Esto le dice a WandB que calcule y guarde los máximos históricos de cada run.
        # Creará de forma independiente: val/MRR (max), val/R@1 (max) y val/R@10 (max)
        # Las originales (val/MRR, etc.) seguirán mostrando el valor de la última epoch.
        wandb.define_metric("val/MRR", summary="max")
        wandb.define_metric("val/R@1", summary="max")
        wandb.define_metric("val/R@10", summary="max")
        # ============================================================

        config = wandb.config
        print(f"\n{'='*60}")
        print(f"Run: {run.name}")
        print(f"  learning_rate : {config.learning_rate}")
        print(f"  batch_size    : {config.batch_size}")
        print(f"  embedding_dim : {config.embedding_dim}")
        print(f"  margin        : {config.margin}")
        print(f"  clip_threshold: {getattr(config, 'clip_threshold', 0.0)}")
        print(f"  epochs        : {config.epochs}")
        print(f"{'='*60}\n")

        model, train_loader, val_loader, criterion, optimizer = make(
            config, device=device
        )

        # train() ya loguea las métricas normales (val/R@1, etc.) en cada epoch
        best_r1_this_run = train(
            model=model,
            loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        if best_r1_this_run > best_r1_global:
            best_r1_global = best_r1_this_run
            os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
            torch.save({
                "epoch":    None,
                "model":    model.state_dict(),
                "best_r1":  best_r1_global,
                "best_r10": None,
            }, BEST_MODEL_PATH)
            print(f"*** NOU MILLOR MODEL del sweep! R@1={best_r1_global:.4f} "
                f"guardat a {BEST_MODEL_PATH} ***")


if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="ResNet18-SBERTcongelat-Triplet-MRR")
    wandb.agent(sweep_id, function=model_pipeline, count=30)