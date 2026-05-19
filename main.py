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


sweep_config = {
    "name": "comics-retrieval-triplet-sweep",
    "method": "bayes",
    "metric": {"name": "val/R@1", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 4,   # mínim 4 epochs abans de poder podar
        "eta": 2,        # poda la meitat dels pitjors runs cada ronda
    },
    "parameters": {
        # Hiperparàmetres d'optimització
        "learning_rate": {"values": [1e-4, 5e-5, 2e-5, 1e-5]},
        "batch_size":    {"values": [64, 128,256,512]},
        # Arquitectura del model
        "embedding_dim": {"values": [512]},
        "margin":        {"values": [0.1, 0.2, 0.3, 0.5]},
        # Epochs: 5 permet desfreeze parcial a epoch 2; 10 a epoch 5
        "epochs":        {"values": [10, 15,20,25,30]},
        # Metadades (no canvien)
        "dataset":       {"value": "ComicsPAP"},
        "architecture":  {"value": "BERT-ResNet"},
    }
}


def model_pipeline():
    """Executa UN run del sweep amb la combinació d'hiperparàmetres assignada per WandB."""
    with wandb.init() as run:
        config = wandb.config

        print(f"\n{'-'*60}")
        print(f"NOU RUN: {run.name}")
        print(f"{'-'*60}")
        print(f"  {'learning_rate':<14} {config.learning_rate}")
        print(f"  {'batch_size':<14} {config.batch_size}")
        print(f"  {'embedding_dim':<14} {config.embedding_dim}")
        print(f"  {'margin':<14} {config.margin}")
        print(f"  {'epochs':<14} {config.epochs}")
        print(f"{'-'*60}\n")

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
        print(f"\n Model final guardat a: {save_path}\n")


if __name__ == "__main__":
    wandb.login()

    sweep_id = wandb.sweep(sweep_config, project="Comics-Retrieval-Sweep")

    # Bayes + early_terminate: WandB decideix quan aturar.
    # Posem count=20 com a límit de seguretat per no gastar recursos infinits.
    wandb.agent(sweep_id, function=model_pipeline, count=20)