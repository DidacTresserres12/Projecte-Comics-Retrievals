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

# ── Millors hiperparàmetres del sweep ──
config_dict = {
    "learning_rate" : 0.0001,
    "batch_size"    : 64,
    "embedding_dim" : 512,
    "margin"        : 0.1,
    "epochs"        : 20,       # més epochs perquè la loss encara baixava
    "freeze_bert"   : True,
    "freeze_resnet" : True,
    "dataset"       : "ComicsPAP",
    "architecture"  : "BERT-ResNet",
}

if __name__ == "__main__":
    wandb.login()

    with wandb.init(project="Comics-Retrieval-Final", config=config_dict, name="best-model-run") as run:
        config = wandb.config

        print(f"\n{'='*60}")
        print(f"Run final amb els millors hiperparàmetres")
        print(f"  learning_rate : {config.learning_rate}")
        print(f"  batch_size    : {config.batch_size}")
        print(f"  embedding_dim : {config.embedding_dim}")
        print(f"  margin        : {config.margin}")
        print(f"  epochs        : {config.epochs}")
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

        # Guardem el model final
        save_path = os.path.join(script_dir, "trained_models", "model_best_final.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model final guardat a: {save_path}")