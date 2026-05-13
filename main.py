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

# ──────────────────────────────────────────────────────────────────────────────
# Configuració del Sweep
#
# Decisions de disseny:
#
# method = "bayes"  (no "grid" ni "random")
#   - "grid" prova TOTES les combinacions: 3 lr × 2 bs × 3 margin × 2 epochs = 36 runs
#     → massa costós. No té sentit provar-ho tot quan l'espai és gran.
#   - "random" prova combinacions aleatòries: millor que grid però "cec" perquè
#     no aprèn de runs anteriors.
#   - "bayes" (Bayesian Optimization) aprèn quines zones de l'espai d'hiperparàmetres
#     donen millors resultats i dirigeix la cerca cap allà. Amb un pressupost limitat
#     de runs, és el mètode que troba millors hiperparàmetres.
#
# early_terminate (Hyperband):
#   Atura aviat els runs que clarament no milloren, estalviant còmput.
#   - min_iter=2: espera almenys 2 epochs abans de poder aturar un run.
#   - eta=2: poda la meitat dels pitjors runs en cada ronda.
#   Compatible amb "bayes" (WandB ho gestiona internament).
#
# batch_size = [64, 128]:
#   - [128, 256] és massa gran per a la majoria de GPUs de VM de 16GB amb ResNet50+BERT.
#     Un batch de 256 imatges (224x224) + BERT ocupa ~12-14GB → OOM probable.
#   - 64 és el mínim raonable per a TripletLoss (necessita diversitat al batch).
#   - 128 és un bon compromís velocitat/diversitat si la GPU ho permet.
#
# epochs = [5, 10]:
#   - 3 epochs és massa poc: amb el 20% del dataset i freeze total, el model
#     gairebé no aprèn (les capes de projecció necessiten més iteracions).
#   - 5 és el mínim útil; 10 permet el desfreeze parcial a la meitat.
#   - [3, 5] de la proposta original no permet el desfreeze (activa a epochs//2,
#     i amb 3 epochs vol dir epoch 1 → massa aviat, amb 5 → epoch 2 → molt just).
#
# num_workers: NO és un hiperparàmetre de model → fora del sweep.
#   Es fixa a 4 directament a utils.py (valor segur per a VMs amb 8 CPU).
#
# ──────────────────────────────────────────────────────────────────────────────
sweep_config = {
    "name": "comics-retrieval-triplet-sweep",
    "method": "bayes",
    "metric": {"name": "val/R@1", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,   # mínim 2 epochs abans de poder podar
        "eta": 2,        # poda la meitat dels pitjors runs cada ronda
    },
    "parameters": {
        # Hiperparàmetres d'optimització
        "learning_rate": {"values": [1e-4, 2e-5, 5e-5]},
        "batch_size":    {"values": [64, 128]},
        # Arquitectura del model
        "embedding_dim": {"values": [512]},
        "margin":        {"values": [0.1, 0.2, 0.3]},
        # Congelació dels encoders (fixos: sempre congelats al inici)
        "freeze_bert":   {"value": True},
        "freeze_resnet": {"value": True},
        # Epochs: 5 permet desfreeze parcial a epoch 2; 10 a epoch 5
        "epochs":        {"values": [5, 10]},
        # Metadades (no canvien)
        "dataset":       {"value": "ComicsPAP"},
        "architecture":  {"value": "BERT-ResNet"},
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
        print(f"  margin        : {config.margin}")
        print(f"  epochs        : {config.epochs}")
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

    sweep_id = wandb.sweep(sweep_config, project="Comics-Retrieval-Sweep")

    # Bayes + early_terminate: WandB decideix quan aturar.
    # Posem count=30 com a límit de seguretat per no gastar recursos infinits.
    wandb.agent(sweep_id, function=model_pipeline, count=30)
