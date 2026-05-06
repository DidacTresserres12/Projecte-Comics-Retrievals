from tqdm.auto import tqdm
import wandb

def train(model, loader, criterion, optimizer, config,device="cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
for epoch in tqdm(range(start_epoch, config.epochs)):
        model.train()
        
        for _, batch in enumerate(loader):
            # Adaptem a l'estructura de diccionari del dataloader de còmics
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            # Cridem a train_batch amb els nous inputs
            loss = train_batch(images, input_ids, mask, model, optimizer, criterion, device=device)
            
            example_ct += len(images)
            batch_ct += 1

            # Report train metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        # 1. Validació de mètriques de rendiment (Recall@K)
        val_metrics = validate_retrieval(model, val_loader, device)
        
        # Log de validació a wandb
        wandb.log({
            "epoch": epoch,
            "val/R@1": val_metrics["R@1"],
            "val/R@5": val_metrics["R@5"]
        }, step=example_ct)

        # Guardat periòdic i del millor model 
        is_best = val_metrics["R@1"] > best_r1
        if is_best:
            best_r1 = val_metrics["R@1"]
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_r1': best_r1,
            'example_ct': example_ct
        }, is_best)


def train_batch(images, input_ids, mask, model, optimizer, criterion, device="cuda"):
    images = images.to(device)
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    
    # Forward pass ➡
    # El model de còmics retorna embeddings de text i imatge
    txt_emb, img_emb = model({'input_ids': input_ids, 'attention_mask': mask}, images)
    
    # En retrieval
    logits = torch.matmul(txt_emb, img_emb.t())
    labels = torch.arange(logits.size(0)).to(device)
    loss = criterion(logits, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


#FUNCIONS AUXILIARS

def validate_retrieval(model, loader, device):
    #Calcula el Recall@K per avaluar el rendiment real.
    model.eval()
    all_txt = []
    all_img = []
    
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
    #Guarda l'estat actual per poder reprendre l'entrenament.
    torch.save(state, "last_checkpoint.pth")
    if is_best:
        torch.save(state, "model_best.pth")
        print(f"⭐ New best model saved (R@1: {state['best_r1']:.4f})")