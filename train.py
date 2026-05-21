from tqdm.auto import tqdm
import wandb
import torch
import os


def get_checkpoint_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "trained_models")
    return os.path.join(save_dir, "model_best.pth")


def unfreeze_last_layers(model):
    """
    Descongela NOMÉS les últimes capes de SentenceBERT i ResNet.

    Per SentenceBERT (MiniLM, 6 capes: encoder.layer.0 a encoder.layer.5):
      Descongelem l'última capa (encoder.layer.5) i el pooler.
      MiniLM té 6 capes en lloc de 12 (BERT base), per tant l'última és la 5.

    Per ResNet50 (model.resnet és nn.Sequential):
      [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool,
      [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4, [8]=avgpool
      Descongelem layer4 (índex 7), els blocs residuals finals.

    Per què no descongelar tot?
      Descongelar tot amb LR alt causa catastrophic forgetting: els pesos
      pre-entrenats que codifiquen coneixement general es destrueixen.
      Les últimes capes codifiquen features d'alt nivell més adaptables
      al domini (còmics) sense perdre les bases de les inferiors.
    """
    print("Descongelant capes finals de SentenceBERT (layer 5) i ResNet (layer4)")

    for name, param in model.sbert.named_parameters():
        if "encoder.layer.5" in name or "pooler" in name:
            param.requires_grad = True

    for param in model.resnet[7].parameters():  # layer4 de ResNet50
        param.requires_grad = True


def save_checkpoint(state, is_best):
    if is_best:
        path = get_checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Millor model guardat a: {path} (R@1: {state['best_r1']:.4f})")


def train(model, loader, val_loader, criterion, optimizer, config, device="cuda") -> float:
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    batch_ct = 0
    best_r1  = 0.0
    best_r10 = 0.0

    # Descongelar les últimes capes des del principi.
    # Amb els encoders 100% congelats les capes de projecció reben sempre
    # els mateixos features → l'espai d'embedding no aprèn res diferenciador.
    unfreeze_last_layers(model)
    print("Capes finals descongelades. Iniciant entrenament.")

    for epoch in tqdm(range(config.epochs)):
        model.train()
        epoch_loss = 0.0

        for _, batch in enumerate(loader):
            images    = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)

            loss = train_batch(images, input_ids, mask, model, optimizer, criterion, device)

            example_ct += len(images)
            batch_ct += 1
            epoch_loss += loss.item()

            if ((batch_ct + 1) % 25) == 0:
                wandb.log(
                    {"train/loss": loss.item(), "train/examples": example_ct},
                    step=batch_ct
                )
                print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

        val_metrics = validate_retrieval(model, val_loader, device)

        # Actualitzar millors mètriques i guardar checkpoint si millora
        is_best = val_metrics["R@1"] > best_r1
        if is_best:
            best_r1  = val_metrics["R@1"]
            best_r10 = val_metrics["R@10"]
            # Guardar el model en el seu millor moment (no al final del run)
            path = get_checkpoint_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "best_r1":   best_r1,
                "best_r10":  best_r10,
            }, path)
            print(f"  ✅ Nou millor model guardat (R@1={best_r1:.4f}, R@10={best_r10:.4f})")

        wandb.log({
            "epoch":            epoch,
            "train/epoch_loss": epoch_loss / len(loader),
            # Mètriques de l'epoch actual
            "val/R@1":          val_metrics["R@1"],
            "val/R@10":         val_metrics["R@10"],
            # Millors mètriques acumulades del run → visibles al WandB summary
            "val/best_R@1":     best_r1,
            "val/best_R@10":    best_r10,
        }, step=batch_ct)

    # Escriure explícitament al summary de WandB → la taula del sweep
    # mostrarà sempre el MÀXIM, no l'últim epoch
    wandb.run.summary["best_R@1"]  = best_r1
    wandb.run.summary["best_R@10"] = best_r10

    # Retornem el millor R@1 d aquest run perque main.py decideixi
    # si es el millor de tot el sweep i guardi el model
    return best_r1


def train_batch(images, input_ids, mask, model, optimizer, criterion, device="cuda"):
    """
    Un pas d'entrenament amb TripletMarginLoss i in-batch negatives.

    Com funciona el mining dels negatius?
    ─────────────────────────────────────
    Concatenem els embeddings de text i imatge en un sol tensor i assignem
    labels iguals a cada parella (text_i, img_i) → label i.

    pytorch_metric_learning mina automàticament tots els triplets vàlids:
      - anchor:   text_i  (o img_i)
      - positive: img_i   (o text_i)  → mateixa label
      - negative: img_j / text_j amb j≠i → label diferent

    Amb batch size N tenim N-1 negatius disponibles per cada anchor.
    Amb batch size gran (128, 256) → molts més triplets → millor gradient.
    """
    model.train()

    text_features = {
        'input_ids':      input_ids.squeeze(1),
        'attention_mask': mask.squeeze(1)
    }
    txt_emb, img_emb = model(text_features, images)

    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)

    N = txt_emb.size(0)

    # Labels: text_i i img_i → label i (parella positiva)
    # Tot el que té label diferent és negatiu automàticament
    indices    = torch.arange(N, device=device)
    embeddings = torch.cat([txt_emb, img_emb], dim=0)   # [2N, emb_dim]
    labels     = torch.cat([indices, indices], dim=0)    # [2N]

    loss = criterion(embeddings, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validate_retrieval(model, loader, device):
    """
    Calcula R@1 i R@10 sobre el conjunt de validació complet.
    """
    model.eval()
    all_txt, all_img = [], []

    with torch.no_grad():
        for batch in loader:
            t, i = model(
                {
                    'input_ids':      batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                },
                batch['image'].to(device)
            )
            all_txt.append(t)
            all_img.append(i)

    txts = torch.nn.functional.normalize(torch.cat(all_txt), p=2, dim=1)
    imgs = torch.nn.functional.normalize(torch.cat(all_img), p=2, dim=1)

    sim  = torch.matmul(txts, imgs.t())
    gt   = torch.arange(len(txts), device=device).view(-1, 1)

    r1  = (sim.topk(1,  dim=1)[1] == gt).any(dim=1).float().mean().item()
    r10 = (sim.topk(min(10, len(txts)), dim=1)[1] == gt).any(dim=1).float().mean().item()

    print(f"  val/R@1={r1:.4f}  val/R@10={r10:.4f}")
    return {"R@1": r1, "R@10": r10}