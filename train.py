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
    Estratègia de descongelació per a ResNet18 + SBERT congelat (recomanació professor).

    ResNet18 (model.resnet és nn.Sequential):
      [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool,
      [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4, [8]=avgpool
      Descongelem layer4 (índex 7), els blocs residuals finals.
      NOTA: ResNet18 vs ResNet50 → mateixa posició de layer4 (índex 7).

    SBERT: es manté completament congelat per decisió del professor.
      Amb SBERT congelat, la projecció de text rep sempre els mateixos
      embeddings fixos i aprèn a alinear-los amb els d'imatge.
      ResNet18 és més petit (512 features vs 2048 de ResNet50), el que
      pot facilitar l'alineació quan el costat de text és fix.
    """
    # SBERT: es deixa tal com estava (congelat si freeze_sbert=True al config)
    sbert_trainable = sum(1 for p in model.sbert.parameters() if p.requires_grad)
    if sbert_trainable == 0:
        print("SBERT completament congelat (recomanació professor)")
    else:
        print(f"SBERT parcialment entrenable ({sbert_trainable} paràmetres)")

    # ResNet18: descongelar layer4 (índex 7 del Sequential)
    print("Descongelant layer4 de ResNet18 (índex 7 del Sequential)")
    for param in model.resnet[7].parameters():   # layer4 de ResNet18
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
    batch_ct   = 0
    best_r1    = 0.0
    best_r10   = 0.0
    best_mrr   = 0.0   # Mean Reciprocal Rank: millor valor vist durant el run

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
            batch_ct   += 1
            epoch_loss += loss.item()

            if ((batch_ct + 1) % 25) == 0:
                wandb.log(
                    {"train/loss": loss.item(), "train/examples": example_ct},
                    step=batch_ct
                )
                print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

        val_metrics = validate_retrieval(model, val_loader, device)

        is_best = val_metrics["R@1"] > best_r1
        if is_best:
            best_r1  = val_metrics["R@1"]
            best_r10 = val_metrics["R@10"]
            path = get_checkpoint_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "best_r1":  best_r1,
                "best_r10": best_r10,
            }, path)
            print(f"  ✅ Nou millor model guardat (R@1={best_r1:.4f}, R@10={best_r10:.4f})")

        # Actualitzem best_mrr independentment (pot millorar en epochs on R@1 no millora)
        if val_metrics["MRR"] > best_mrr:
            best_mrr = val_metrics["MRR"]

        # Loguem les mètriques de l'epoch actual (les corbes de val al llarg del temps)
        wandb.log({
            "epoch":            epoch,
            "train/epoch_loss": epoch_loss / len(loader),
            "val/R@1":          val_metrics["R@1"],
            "val/R@10":         val_metrics["R@10"],
            "val/MRR":          val_metrics["MRR"],
        }, step=batch_ct)

    # Guardem els MILLORS valors de tot el run al summary de WandB.
    # Això fa que a la taula del sweep apareguin 3 columnes amb el màxim
    # assolit (no el de l'últim epoch): best_R@1, best_R@10, best_MRR.
    wandb.run.summary["val/best_R@1"]  = best_r1
    wandb.run.summary["val/best_R@10"] = best_r10
    wandb.run.summary["val/best_MRR"]  = best_mrr

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
    Calcula R@1, R@10 i MRR (Mean Reciprocal Rank) sobre el conjunt de validació.

    MRR = mitjana de 1/rank de la imatge correcta per a cada consulta.
      - Si la imatge correcta és la 1a → contribueix 1.0
      - Si és la 2a → 0.5, la 3a → 0.33, etc.
    MRR és sensible a si el model "quasi encerta": un model que sempre posa
    la solució en posició 2 tindrà MRR≈0.5 però R@1=0. Complementa R@K.
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

    # MRR: per a cada consulta, trobem el rang (1-based) de la imatge correcta
    # argsort descendent → posicions de menor a major similitud (rang 0 = millor)
    ranks = (sim.argsort(dim=1, descending=True) == gt).nonzero(as_tuple=False)[:, 1]
    mrr   = (1.0 / (ranks.float() + 1)).mean().item()

    print(f"  val/R@1={r1:.4f}  val/R@10={r10:.4f}  val/MRR={mrr:.4f}")
    return {"R@1": r1, "R@10": r10, "MRR": mrr}