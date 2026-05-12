from tqdm.auto import tqdm
import wandb
import torch
import os


def get_checkpoint_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "trained_models")
    return os.path.join(save_dir, "model_best.pth")


def unfreeze_layers(model):
    print("Descongelant BERT i ResNet per al fine-tuning final")
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(state, is_best):
    if is_best:
        path = get_checkpoint_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Mejor modelo guardado en: {path} (R@1: {state['best_r1']:.4f})")


def train(model, loader, val_loader, criterion, optimizer, config, device="cuda"):
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    batch_ct = 0
    best_r1 = 0.0
    start_epoch = 0

    # --- Resume from checkpoint if it exists ---
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

    for epoch in tqdm(range(start_epoch, config.epochs)):
        # --- NOVA LÒGICA DE DESCONGELACIÓ ---
        if epoch == 5:
            unfreeze_layers(model)
            # Opcional: Pots reduir el LR aquí perquè el fine-tuning no sigui brusc
            for param_group in optimizer.param_groups:
                # Reduïm el learning rate per al fine-tuning
                param_group['lr'] = config.learning_rate / 10

        model.train()

        for _, batch in enumerate(loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            loss = train_batch(images, input_ids, mask, model, optimizer, criterion, device=device)

            example_ct += len(images)
            batch_ct += 1

            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        # Validación
        val_metrics = validate_retrieval(model, val_loader, device)

        wandb.log({
            "epoch": epoch,
            "val/R@1": val_metrics["R@1"],
            "val/R@10": val_metrics["R@10"]
        }, step=epoch) # Fiquem epoch a l'eix x de WandB per veure millor l'evolució al llarg de les epochs

        # Guardado
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
    txt_emb, img_emb = model({'input_ids': input_ids.squeeze(1), 'attention_mask': mask.squeeze(1)}, images)

    # Normalizamos
    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)

    # Preparamos etiquetas para la librería
    indices = torch.arange(txt_emb.size(0)).to(device)

    # Concatenamos para que la loss compare todo el batch
    embeddings = torch.cat([txt_emb, img_emb], dim=0)
    labels = torch.cat([indices, indices], dim=0)

    # Calculamos la Loss usando la nueva librería
    loss = criterion(embeddings, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def validate_retrieval(model, loader, device):
    model.eval()
    all_txt, all_img = [], []

    with torch.no_grad():   
        for batch in loader:
            t, i = model({
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }, batch['image'].to(device))
            all_txt.append(t)
            all_img.append(i)

    txts = torch.cat(all_txt)
    imgs = torch.cat(all_img)
    sim = torch.matmul(txts, imgs.t())

    correct = torch.arange(len(txts)).to(device).view(-1, 1)
    r1 = (sim.topk(1, dim=1)[1] == correct).any(dim=1).float().mean().item()
    r10 = (sim.topk(10, dim=1)[1] == correct).any(dim=1).float().mean().item()

    return {"R@1": r1, "R@10": r10}