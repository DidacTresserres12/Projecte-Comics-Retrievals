"""
compute_clip_scores.py
======================
Calcula el CLIP score (similitud cosinus imatge-text) per a cada mostra
de tots els splits del dataset ComicsPAP i guarda els resultats a:

    clip_scores/clip_scores_train.json
    clip_scores/clip_scores_validation.json
    clip_scores/clip_scores_test.json

Cada fitxer té format:
    { "0": 0.2341, "1": 0.1823, "2": 0.3012, ... }

on la clau és l'índex de la mostra i el valor és el CLIP score.

Execució:
    python compute_clip_scores.py
    python compute_clip_scores.py --splits validation
    python compute_clip_scores.py --batch_size 64
"""

import io
import os
import sys
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

DATASET_PATH = "/home/datasets/COMIC-PAP"
OUTPUT_DIR   = "clip_scores"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 32


def get_image_and_caption(sample):
    sol_idx = sample["solution_index"]
    if sol_idx < 0:
        sol_idx = 0
    opt = sample["options"][sol_idx]
    img_bytes = opt["bytes"] if isinstance(opt, dict) else opt
    image   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    caption = sample.get("previous_panel_caption", "") or ""
    return image, str(caption)


@torch.no_grad()
def compute_scores_batch(clip_model, clip_processor, device, images, captions):
    # Truncar captions a 77 tokens (límit de CLIP)
    inputs = clip_processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)
    outputs  = clip_model(**inputs)
    img_norm = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
    # Similitud cosinus per a cada parell (diagonal)
    scores = (img_norm * txt_norm).sum(dim=-1).cpu().tolist()
    return scores


def process_split(split_name, clip_model, clip_processor, device, batch_size):
    print(f"\n{'='*50}")
    print(f"Processant split: {split_name}")
    split_data = load_dataset(DATASET_PATH, split=split_name)
    print(f"  {len(split_data)} mostres")

    scores_dict = {}
    errors = 0

    # Processar en batches
    images_batch   = []
    captions_batch = []
    indices_batch  = []

    for i, sample in enumerate(tqdm(split_data, desc=f"  {split_name}")):
        try:
            image, caption = get_image_and_caption(sample)
            images_batch.append(image)
            captions_batch.append(caption[:500])  # limitar longitud
            indices_batch.append(i)
        except Exception as e:
            errors += 1
            scores_dict[str(i)] = 0.0
            continue

        if len(images_batch) >= batch_size:
            scores = compute_scores_batch(
                clip_model, clip_processor, device,
                images_batch, captions_batch
            )
            for idx, score in zip(indices_batch, scores):
                scores_dict[str(idx)] = round(float(score), 4)
            images_batch   = []
            captions_batch = []
            indices_batch  = []

    # Últim batch parcial
    if images_batch:
        scores = compute_scores_batch(
            clip_model, clip_processor, device,
            images_batch, captions_batch
        )
        for idx, score in zip(indices_batch, scores):
            scores_dict[str(idx)] = round(float(score), 4)

    # Estadístiques
    vals = list(scores_dict.values())
    vals_nonzero = [v for v in vals if v > 0]
    print(f"\n  Estadístiques de CLIP score per '{split_name}':")
    print(f"    Mostres processades: {len(vals_nonzero)} / {len(split_data)}")
    print(f"    Errors: {errors}")
    if vals_nonzero:
        import statistics
        print(f"    Min:    {min(vals_nonzero):.4f}")
        print(f"    Max:    {max(vals_nonzero):.4f}")
        print(f"    Mitjà:  {statistics.mean(vals_nonzero):.4f}")
        print(f"    Mediana:{statistics.median(vals_nonzero):.4f}")

        # Distribució per buckets
        buckets = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        print(f"\n    Distribució (% de mostres per sota del threshold):")
        for thr in buckets:
            count = sum(1 for v in vals_nonzero if v < thr)
            pct   = 100 * count / len(vals_nonzero)
            print(f"      < {thr:.2f}: {count:6d} mostres ({pct:.1f}%) → es descartarien")

    return scores_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",     nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--output_dir", type=str,   default=OUTPUT_DIR)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Dispositiu: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nCarregant CLIP ({CLIP_MODEL})...")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()
    print("  CLIP carregat")

    for split in args.splits:
        scores = process_split(split, clip_model, clip_processor, device, args.batch_size)

        out_path = os.path.join(args.output_dir, f"clip_scores_{split}.json")
        with open(out_path, "w") as f:
            json.dump(scores, f)
        print(f"\n  Scores guardats a: {out_path}")

    print(f"\nFet! Tots els scores guardats a: {args.output_dir}/")
    print("Ara podeu executar main.py amb --clip_threshold per filtrar mostres.")


if __name__ == "__main__":
    main()
