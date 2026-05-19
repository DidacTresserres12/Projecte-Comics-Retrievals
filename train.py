from tqdm.auto import tqdm
import wandb
import torch
import os


def save_checkpoint(state, checkpoint_path):
    """Guarda el model si és el millor fins ara (R@1 màxim)."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    print(f" Millor model guardat (R@1: {state['best_r1']:.4f}) → {checkpoint_path}")


def train(model, loader, val_loader, criterion, optimizer, config, device="cuda"):
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0
    batch_ct = 0
    best_r1 = 0.0

    # Ruta del millor model: dins trained_models/ al mateix nivell que el projecte.
    save_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "trained_models")
    )
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "model_best.pth")

    print("  Iniciant entrenament des de zero...\n")

    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="ep"):

        model.train()

        epoch_loss = 0.0
        for _, batch in enumerate(loader):
            images    = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)

            loss = train_batch(images, input_ids, mask, model, optimizer, criterion, device=device)

            example_ct += len(images)
            batch_ct   += 1
            epoch_loss += loss.item()

            if ((batch_ct + 1) % 25) == 0:
                wandb.log({"train/loss": loss.item(), "train/examples": example_ct}, step=batch_ct)
                print(f" [Batch {batch_ct:>4}] Loss: {loss:.4f}  (exemples vistos: {example_ct:>5})")

        val_metrics = validate_retrieval(model, val_loader, device)

        wandb.log({
            "epoch":            epoch,
            "train/epoch_loss": epoch_loss / len(loader),
            "val/R@1":          val_metrics["R@1"],
            "val/R@10":         val_metrics["R@10"],
        }, step=batch_ct)

        if val_metrics["R@1"] > best_r1:
            best_r1 = val_metrics["R@1"]
            save_checkpoint({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_r1':          best_r1,
            }, checkpoint_path)


def train_batch(images, input_ids, mask, model, optimizer, criterion, device="cuda"):
    model.train()

    txt_emb, img_emb = model(
        {'input_ids': input_ids.squeeze(1), 'attention_mask': mask.squeeze(1)},
        images
    )

    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)

    indices    = torch.arange(txt_emb.size(0), device=device)
    embeddings = torch.cat([txt_emb, img_emb], dim=0)
    labels     = torch.cat([indices, indices], dim=0)

    loss = criterion(embeddings, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def validate_retrieval(model, loader, device):
    """
    Calcula R@1 i R@10 sobre tot el conjunt de validació.

    Construïm la matriu de similaritat NxN entre tots els embeddings de text
    i d'imatge del val set. Per cada text i, comprovem si la imatge correcta
    (índex i) apareix entre els top-K resultats.
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

    sim     = torch.matmul(txts, imgs.t())
    correct = torch.arange(len(txts), device=device).view(-1, 1)

    r1  = (sim.topk(1,                  dim=1)[1] == correct).any(dim=1).float().mean().item()
    r10 = (sim.topk(min(10, len(txts)), dim=1)[1] == correct).any(dim=1).float().mean().item()

    print(f" Validació   ->   R@1: {r1:.4f}   |   R@10: {r10:.4f}")
    return {"R@1": r1, "R@10": r10}