import wandb
import torch


def test(model, test_loader, device="cuda"):
    """
    Evaluate the retrieval model on the test set.
    Computes R@1, R@5 and R@10 (text-to-image retrieval).
    Logs results to Weights & Biases.
    """
    model.eval()
    all_txt, all_img = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)

            txt_emb, img_emb = model(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                images
            )
            all_txt.append(txt_emb)
            all_img.append(img_emb)

    txts = torch.cat(all_txt)   # [N, embedding_dim]
    imgs = torch.cat(all_img)   # [N, embedding_dim]

    # Similarity matrix [N, N]
    sim = torch.matmul(txts, imgs.t())

    ground_truth = torch.arange(len(txts), device=device).view(-1, 1)

    r1  = (sim.topk(1,  dim=1)[1] == ground_truth).any(dim=1).float().mean().item()
    r5  = (sim.topk(5,  dim=1)[1] == ground_truth).any(dim=1).float().mean().item()
    r10 = (sim.topk(10, dim=1)[1] == ground_truth).any(dim=1).float().mean().item()

    print(f"Test R@1:  {r1:.4f}")
    print(f"Test R@5:  {r5:.4f}")
    print(f"Test R@10: {r10:.4f}")

    wandb.log({
        "test/R@1":  r1,
        "test/R@5":  r5,
        "test/R@10": r10,
    })

    return {"R@1": r1, "R@5": r5, "R@10": r10}