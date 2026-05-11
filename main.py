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

# Reproducibilitat
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Configuració del Sweep (exploració exhaustiva)
# ─────────────────────────────────────────────
sweep_config = {
    "name": "comics-retrieval-sweep",
    "method": "grid",   # prova TOTES les combinacions
    "metric": {
        "name": "val/R@1",
        "goal": "maximize"
    },
    "parameters": {
        # Hiperparàmetres d'optimització
        "learning_rate": {
            "values": [1e-4, 2e-5, 5e-5]
        },
        "batch_size": {
            "values": [32, 64]
        },
        # Arquitectura del model
        "embedding_dim": {
            "values": [256, 512]
        },
        # Congelació dels encoders
        "freeze_bert": {
            "values": [True, False]
        },
        "freeze_resnet": {
            "values": [True, False]
        },
        # Fixos — no són hiperparàmetres a explorar
        "epochs": {
            "value": 10
        },
        "dataset": {
            "value": "ComicsPAP"
        },
        "architecture": {
            "value": "BERT-ResNet"
        },
    }
}


def model_pipeline():
    """
    Funció que executa UN run del sweep.
    WandB crida aquesta funció per a cada combinació de paràmetres.
    """
    with wandb.init() as run:
        config = wandb.config
        print(f"\n{'='*60}")
        print(f"Run: {run.name}")
        print(f"  learning_rate : {config.learning_rate}")
        print(f"  batch_size    : {config.batch_size}")
        print(f"  embedding_dim : {config.embedding_dim}")
        print(f"  freeze_bert   : {config.freeze_bert}")
        print(f"  freeze_resnet : {config.freeze_resnet}")
        print(f"{'='*60}\n")

        model, train_loader, val_loader, criterion, optimizer = make(config, device=device)

        train(
            model=model,
            loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
        )

        # Guardem el model d'aquest run
        save_dir = os.path.abspath(os.path.join(script_dir, "..", "trained_models"))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"model_{run.name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model guardat a: {save_path}")


if __name__ == "__main__":
    wandb.login()

    # Creem el sweep
    sweep_id = wandb.sweep(sweep_config, project="Comics-Retrieval-Sweep")

    # Llancem tots els runs del sweep en aquest agent
    # count=None → executa TOTES les combinacions (grid)
    wandb.agent(sweep_id, function=model_pipeline, count=None)