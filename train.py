from tqdm.auto import tqdm
import wandb
import torch

def train(model, loader, val_loader, criterion, optimizer, config, device="cuda"):
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0  
    batch_ct = 0
    best_r1 = 0.0
    for epoch in tqdm(range(config.epochs)):
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
        
        # Log de validación a wandb
        wandb.log({
            "epoch": epoch,
            "val/R@1": val_metrics["R@1"],
            "val/R@5": val_metrics["R@5"]
        }, step=example_ct)

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
    # Forward pass
    txt_emb, img_emb = model({'input_ids': input_ids, 'attention_mask': mask}, images)
    
    # Similitud y pérdida
    logits = torch.matmul(txt_emb, img_emb.t())
    labels = torch.arange(logits.size(0)).to(device)
    loss = criterion(logits, labels)
    
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
    r5 = (sim.topk(5, dim=1)[1] == correct).any(dim=1).float().mean().item()
    
    return {"R@1": r1, "R@5": r5}

def save_checkpoint(state, is_best):
    if is_best:
        torch.save(state, "model_best.pth")
        print(f"Mejor modelo guardado (R@1: {state['best_r1']:.4f})")