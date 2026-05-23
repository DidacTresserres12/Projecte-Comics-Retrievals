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
    "name":   "resnet50-sbert-congelat-v2",   # nom visible al dashboard de WandB
    "method": "bayes",
    "metric": {"name": "val/best_R@1", "goal": "maximize"},
    # ── Early stopping més tolerant ──────────────────────────────────────────
    # Abans: min_iter=5, eta=2  → massa agressiu, matava runs de 256 que
    # triguen més epochs a convergir.
    # Ara:   min_iter=8, eta=3  → dona més marge per recuperar-se de baixades
    # temporals i permet que batch sizes grans (256, 512) arribin a convergir.
    "early_terminate": {
        "type":     "hyperband",
        "min_iter": 8,
        "eta":      3,
    },
    "parameters": {
        "learning_rate":   {"values": [1e-4, 5e-5, 2e-5, 1e-5]},
        # Afegim batch_size 512 (recomanació prof. per a còmics: batch gran = més negatius)
        "batch_size":      {"values": [64, 128, 256, 512]},
        "embedding_dim":   {"values": [256, 512]},
        "margin":          {"values": [0.1, 0.2, 0.3, 0.4]},
        # Ampliem rang de clip_threshold per trobar el filtratge òptim
        "clip_threshold":  {"values": [0.0, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.32]},
        "freeze_sbert":    {"value": True},    # SBERT sempre congelat (prof.)
        "freeze_resnet":   {"value": True},    # ResNet50 layer4 es descongela a train.py
        "epochs":          {"values": [10, 15, 20]},
        "dataset":         {"value": "ComicsPAP"},
        "architecture":    {"value": "ResNet50-SBERTcongelat"},
    }
}

# R@1 millor global entre TOTS els runs del sweep
best_r1_global = 0.0


def model_pipeline():
    global best_r1_global

    with wandb.init() as run:
        # ── Definim les mètriques del summary ───────────────────────────────
        # IMPORTANT: NO posem summary="max" a val/R@1, val/R@10, val/MRR
        # perquè això fa que WandB sobreescrigui el valor "current" de la
        # taula amb el màxim històric, confondent el valor actual amb el millor.
        #
        # En canvi, loguem les BEST com a claus separades (val/best_R@1, etc.)
        # que s'escriuen directament al summary des de train.py.
        # Així la taula del sweep mostra:
        #   val/R@1     → valor de l'ÚLTIM epoch (corba temporal)
        #   val/best_R@1 → màxim assolit durant tot el run (summary)
        wandb.define_metric("val/R@1",  summary="none")
        wandb.define_metric("val/R@10", summary="none")
        wandb.define_metric("val/MRR",  summary="none")

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
            }, BEST_MODEL_PATH)
            print(f"*** NOU MILLOR MODEL del sweep! R@1={best_r1_global:.4f} "
                f"guardat a {BEST_MODEL_PATH} ***")


if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="ResNet50-SBERTcongelat-Triplet-v2")
    wandb.agent(sweep_id, function=model_pipeline, count=50)