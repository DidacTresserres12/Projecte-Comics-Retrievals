import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from models.models import Baseline


def load_model(model_path, embedding_dim=512, device="cpu"):
    """
    Carrega un model Baseline entrenat prèviament des d'un fitxer .pth
    """
    model = Baseline(embedding_dim=embedding_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # El checkpoint pot ser el state_dict directament o un dict niuat
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model carregat des de: {model_path}")
    return model


def test_complet(model, test_loader, device="cuda"):
    """
    Avalua el model en tot el conjunt de test (R@1 i R@10).
    """
    model.eval()
    all_txt, all_img = [], []

    print("Calculant representacions del dataset de test...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            txt_emb, img_emb = model(
                {'input_ids': batch['input_ids'].to(device),
                 'attention_mask': batch['attention_mask'].to(device)},
                batch['image'].to(device)
            )
            all_txt.append(txt_emb.cpu())
            all_img.append(img_emb.cpu())

    txts = torch.cat(all_txt)
    imgs = torch.cat(all_img)

    sim = torch.matmul(txts, imgs.t())
    ground_truth = torch.arange(len(txts)).view(-1, 1)

    r1  = (sim.topk(1,  dim=1)[1] == ground_truth).any(dim=1).float().mean().item()
    r10 = (sim.topk(10, dim=1)[1] == ground_truth).any(dim=1).float().mean().item()

    print(f"\n--- Resultats de l'Avaluació ---")
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@10: {r10:.4f}")
    return r1, r10


def demo_personalitzada(model, image_path, textos_candidats, device="cpu"):
    """
    Funció de DEMO: dona una imatge i unes frases, el model tria la millor.
    """
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    encoding = model.bert_tokenizer(
        textos_candidats, padding=True, truncation=True, return_tensors="pt"
    )
    text_features = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        txt_emb, img_emb = model(
            text_features,
            img_tensor.expand(len(textos_candidats), -1, -1, -1)
        )

    similituds = F.cosine_similarity(txt_emb, img_emb)
    millor_index = similituds.argmax().item()

    print(f"\n--- Resultat de la DEMO ---")
    for i, text in enumerate(textos_candidats):
        print(f"Frase: '{text}' -> Similitud: {similituds[i].item():.4f}")
    print(f"\nLa frase que millor descriu la imatge és: '{textos_candidats[millor_index]}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test i demo del model Comics Retrieval")

    parser.add_argument("--model_path",    type=str, required=True,
                        help="Ruta al fitxer .pth del model entrenat")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Dimensió de l'embedding (per defecte: 512)")
    parser.add_argument("--device",        type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Dispositiu de càlcul")

    # Mode demo
    parser.add_argument("--image", type=str, default=None,
                        help="[Demo] Ruta a la imatge d'exemple")
    parser.add_argument("--texts", type=str, nargs="+", default=None,
                        help="[Demo] Textos candidats separats per espai")

    # Mode avaluació completa
    parser.add_argument("--eval", action="store_true",
                        help="[Eval] Avalua sobre el dataset de test complet")

    args = parser.parse_args()

    model = load_model(args.model_path, args.embedding_dim, args.device)

    # Mode demo
    if args.image and args.texts:
        demo_personalitzada(model, args.image, args.texts, device=args.device)

    # Mode avaluació completa
    if args.eval:
        from datasets import load_dataset
        from dataloaders.dataset import ComicsPAPDataset

        # Ruta base del dataset en el servidor de la UAB
        base_path = "/home/datasets/COMIC-PAP/datasets/VLR-CVC___comics_pap/caption_relevance/0.0.0/0fde0c1ff630e112937ebca83e2d96a4f533ddfa/"
        
        # Lista de shards del dataset de test
        test_files = [
            base_path + "comics_pap-test-00000-of-00003.arrow",
            base_path + "comics_pap-test-00001-of-00003.arrow",
            base_path + "comics_pap-test-00002-of-00003.arrow"
        ]

        # Cargamos los ficheros directamente para evitar problemas de cuota de disco
        raw_dataset = load_dataset("arrow", data_files={'test': test_files})
        test_data   = ComicsPAPDataset(raw_dataset['test'], model.bert_tokenizer)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=32, shuffle=False, num_workers=4
        )
        test_complet(model, test_loader, device=args.device)

    if not args.image and not args.eval:
        print("\nIndica --image + --texts per al mode demo, o --eval per avaluar el dataset complet.")
        parser.print_help()