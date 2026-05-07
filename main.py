import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
# Esto detecta la carpeta donde está main.py y la añade al buscador de Python
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from utils.utils import make
from train import train
from tqdm.auto import tqdm
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg: dict) -> None:
    # Iniciamos el proyecto en Weights & Biases
    with wandb.init(project="Comics-Retrieval-Final", config=cfg):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer = make(config, device=device)
        train(
            model=model, 
            loader=train_loader, 
            val_loader=test_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            config=config, 
            device=device
        )
        # Guardamos el resultado final fuera de la carpeta del proyecto
        # Se guarda en ../trained_models/ relativo a este script
        save_dir = os.path.abspath(os.path.join(script_dir, "..", "trained_models"))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "modelo_entrenado_comics.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Entrenamiento finalizado y modelo guardado en: {save_path}")

    return model

if __name__ == "__main__":
    wandb.login()

    # Parámetros del experimento
    config = dict(
        epochs=10,
        batch_size=32,
        learning_rate=2e-5,
        dataset="ComicsPAP",
        architecture="BERT-ResNet"
    )

    # Lanzamos el proceso
    model = model_pipeline(config)