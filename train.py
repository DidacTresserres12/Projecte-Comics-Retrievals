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
    Descongela NOMÉS les capes de projecció (text_projection i image_projection)
    i les últimes capes de BERT (layer 11) i ResNet (layer4).

    Per què no descongelar tot?
    - BERT i ResNet ja venen pre-entrenats amb molt bon coneixement general.
    - Descongelar tot de cop amb un LR alt destrueix els pesos (catastrophic forgetting).
    - Descongelar només les últimes capes és suficient per adaptar-se al domini
      de còmics sense perdre el coneixement pre-entrenat.
    - Les capes de projecció ja s'entrenen des del principi (no estan congelades).
    """
    print("Descongelant capes finals de BERT (layer 11) i ResNet (layer4) per al fine-tuning")

    # Descongela l'última capa de BERT (encoder.layer.11 + pooler)
    for name, param in model.bert.named_parameters():
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True

    # Descongela l'últim bloc de ResNet (layer4, els últims ~3 blocs residuals)
    # model.resnet és un nn.Sequential dels fills del ResNet50 original:
    # [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool,
    # [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4, [8]=avgpool
    for param in model.resnet[7].parameters():  # layer4
        param.requires_grad = True


def save_checkpoint(state, is_best):
    if is_best:
        path = get_checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Millor model guardat a: {path} (R@1: {state['best_r1']:.4f})")


def train(model, loader, val_loader, criterion, optimizer, config, device="cuda"):
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    batch_ct = 0
    best_r1 = 0.0
    start_epoch = 0

    # --- Reprendre des de checkpoint si existeix ---
    checkpoint_path = get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint trobat a {checkpoint_path}, reprenent l'entrenament...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_r1 = checkpoint['best_r1']
        start_epoch = checkpoint['epoch']
        print(f"Reprenent des de l'epoch {start_epoch} amb best R@1: {best_r1:.4f}")
    else:
        print("No s'ha trobat cap checkpoint, iniciant entrenament des de zero.")

    # Calculem a quina epoch descongelar:
    # A la meitat del training, però com a mínim a l'epoch 2 (si hi ha epochs suficients).
    # Amb epochs=[3,5]: descongelar a epoch 1 (de 3) o epoch 2 (de 5).
    # Amb epochs=10 (futur): descongelar a epoch 5.
    # Si només hi ha 3 epochs, mai descongelarem (no val la pena, massa poc temps).
    unfreeze_at = config.epochs // 2  # meitat del training

    for epoch in tqdm(range(start_epoch, config.epochs)):

        # Descongelar a la meitat del training (només si hi ha prou epochs)
        if config.epochs >= 4 and epoch == unfreeze_at:
            unfreeze_last_layers(model)
            # Reduïm LR x10 per al fine-tuning: canvis petits per no destruir els pesos
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate / 10
            print(f"  [Epoch {epoch}] Fine-tuning activat. LR → {config.learning_rate / 10:.2e}")

        model.train()

        epoch_loss = 0.0
        for _, batch in enumerate(loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            loss = train_batch(images, input_ids, mask, model, optimizer, criterion, device=device)

            example_ct += len(images)
            batch_ct += 1
            epoch_loss += loss.item()

            if ((batch_ct + 1) % 25) == 0:
                # Usem step=batch_ct (monotònicament creixent) per evitar conflictes WandB
                wandb.log({"train/loss": loss.item(), "train/examples": example_ct}, step=batch_ct)
                print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

        # Validació al final de cada epoch
        val_metrics = validate_retrieval(model, val_loader, device)

        # Tots els logs de l'epoch en una sola crida a wandb.log,
        # usant step=batch_ct (el mateix eix x que la loss de train).
        # Això evita el bug de WandB on barrejar step=example_ct i step=epoch
        # feia que els logs de validació apareguessin tots al mateix punt.
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": epoch_loss / len(loader),
            "val/R@1":  val_metrics["R@1"],
            "val/R@10": val_metrics["R@10"],
        }, step=batch_ct)

        is_best = val_metrics["R@1"] > best_r1
        if is_best:
            best_r1 = val_metrics["R@1"]
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_r1': best_r1,
            }, is_best)


def train_batch(images, input_ids, mask, model, optimizer, criterion, device="cuda"):
    model.train()
    txt_emb, img_emb = model(
        {'input_ids': input_ids.squeeze(1), 'attention_mask': mask.squeeze(1)},
        images
    )

    # Els embeddings ja surten normalitzats del model (F.normalize al encode_text/encode_image),
    # però tornem a normalitzar per seguretat per si el model es modifica en el futur.
    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)

    indices = torch.arange(txt_emb.size(0), device=device)
    embeddings = torch.cat([txt_emb, img_emb], dim=0)
    labels = torch.cat([indices, indices], dim=0)

    loss = criterion(embeddings, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def validate_retrieval(model, loader, device):
    """
    Calcula R@1 i R@10 sobre TOT el conjunt de validació.

    Com funciona:
    1. Extraiem tots els embeddings de text i d'imatge del val set.
    2. Calculem la matriu de similaritat cosinus (NxN).
    3. Per cada text i, mirem si la imatge correcta (índex i) apareix
       entre els top-K resultats ordenats per similaritat.

    Per què dona sempre el mateix valor sense la normalització?
    - El model ja normalitza internament, però encode_text té un bug:
        `with torch.no_grad() if self.bert.training is False else torch.enable_grad()`
      Quan fem model.eval(), self.bert.training=False → sempre entra al no_grad,
      cosa que és correcta per a validació. Però si al train no es fa model.train()
      correctament, BERT no actualitza pesos i els embeddings no canvien entre epochs.
    - A més, si wandb.log barreja steps no monotònicament creixents entre la loss
      de train (step=example_ct, milers) i val (step=epoch, petits), WandB descarta
      o col·lapsa els punts de val → apareix com un sol valor constant al gràfic.
    """
    model.eval()
    all_txt, all_img = [], []

    with torch.no_grad():
        for batch in loader:
            t, i = model(
                {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                },
                batch['image'].to(device)
            )
            all_txt.append(t)
            all_img.append(i)

    # Els embeddings ja surten normalitzats del model; ho fem explícit per robustesa
    txts = torch.nn.functional.normalize(torch.cat(all_txt), p=2, dim=1)
    imgs = torch.nn.functional.normalize(torch.cat(all_img), p=2, dim=1)

    # Matriu de similaritat cosinus: shape [N_val, N_val]
    sim = torch.matmul(txts, imgs.t())

    correct = torch.arange(len(txts), device=device).view(-1, 1)
    r1  = (sim.topk(1,               dim=1)[1] == correct).any(dim=1).float().mean().item()
    r10 = (sim.topk(min(10, len(txts)), dim=1)[1] == correct).any(dim=1).float().mean().item()

    print(f"  val/R@1={r1:.4f}  val/R@10={r10:.4f}")
    return {"R@1": r1, "R@10": r10}
