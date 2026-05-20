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
    "name":   "comics-retrieval-sbert-triplet-sweep",
    "method": "bayes",
    "metric": {"name": "val/R@1", "goal": "maximize"},
    "early_terminate": {
        "type":     "hyperband",
        "min_iter": 2,
        "eta":      2,
    },
    "parameters": {
        "learning_rate": {"values": [1e-4, 2e-5, 5e-5]},
        "batch_size":    {"values": [64, 128, 256]},
        "embedding_dim": {"values": [512]},
        "margin":        {"values": [0.1, 0.2, 0.3]},
        "freeze_sbert":  {"value": True},
        "freeze_resnet": {"value": True},
        "epochs":        {"values": [5, 10]},
        "dataset":       {"value": "ComicsPAP"},
        "architecture":  {"value": "SentenceBERT-ResNet50-Triplet"},
    }
}

# R@1 millor global entre TOTS els runs del sweep
# (variable a nivell de mòdul per comparar entre runs)
best_r1_global = 0.0


def model_pipeline():
    global best_r1_global

    with wandb.init() as run:
        config = wandb.config
        print(f"\n{'='*60}")
        print(f"Run: {run.name}")
        print(f"  learning_rate : {config.learning_rate}")
        print(f"  batch_size    : {config.batch_size}")
        print(f"  embedding_dim : {config.embedding_dim}")
        print(f"  margin        : {config.margin}")
        print(f"  epochs        : {config.epochs}")
        print(f"{'='*60}\n")

        model, train_loader, val_loader, criterion, optimizer = make(
            config, device=device
        )

        # train() ja logueja totes les mètriques a wandb (val/R@1, val/R@10,
        # train/loss...) per a CADA run. Només guarda localment el millor.
        best_r1_this_run = train(
            model=model,
            loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        # Guardem el model localment NOMÉS si és el millor de tot el sweep
        if best_r1_this_run > best_r1_global:
            best_r1_global = best_r1_this_run
            os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** NOU MILLOR MODEL del sweep! R@1={best_r1_global:.4f} "
                  f"guardat a {BEST_MODEL_PATH} ***")


if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="Comics-Retrieval-millores")
    wandb.agent(sweep_id, function=model_pipeline, count=30)
