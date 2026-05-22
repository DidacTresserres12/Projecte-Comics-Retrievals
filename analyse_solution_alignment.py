"""
analyse_solution_alignment.py
==============================
Analitza si el CLIP score identifica correctament la imatge solució.

Per cada mostra:
  1. Agafa TOTS els paràgrafs de la caption
  2. Calcula el CLIP score de cada paràgraf amb CADA opció d'imatge
  3. Fa la mitjana dels scores de tots els paràgrafs per cada imatge
  4. Comprova si la imatge amb score màxim és el solution_index

Guarda a 'alignment_analysis/':
  - mismatches_{split}.json  → mostres on CLIP no identifica la solució
  - summary_{split}.txt      → resum estadístic
  - all_results_{split}.json → tots els resultats (opcional amb --save_all)

Execució:
  python analyse_solution_alignment.py
  python analyse_solution_alignment.py --splits train
  python analyse_solution_alignment.py --splits validation --save_all
"""

import io
import os
import json
import argparse
import statistics
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

DATASET_PATH = "/home/datasets/COMIC-PAP"
OUTPUT_DIR   = "alignment_analysis"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 64


# ─────────────────────────────────────────────────────────────────────────────
# Utilitats
# ─────────────────────────────────────────────────────────────────────────────

def load_image(opt):
    """Carrega una opció d'imatge com a PIL RGB."""
    if isinstance(opt, Image.Image):
        return opt.convert("RGB")
    img_bytes = opt["bytes"] if isinstance(opt, dict) else opt
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def get_paragraphs(sample):
    """Retorna la llista de paràgrafs no buits de la caption."""
    caption = sample.get("previous_panel_caption", "") or ""
    return [p.strip() for p in caption.split("\n") if p.strip()]


@torch.no_grad()
def compute_scores_batch(clip_model, clip_processor, device, images, captions):
    """CLIP score per cada parell (imatge_i, caption_i) → llista de floats."""
    inputs = clip_processor(
        text=captions, images=images,
        return_tensors="pt", padding=True,
        truncation=True, max_length=77
    ).to(device)
    out      = clip_model(**inputs)
    img_norm = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_norm = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    return (img_norm * txt_norm).sum(dim=-1).cpu().tolist()


def compute_paragraph_image_matrix(clip_model, clip_processor, device,
                                   images, paragraphs, batch_size):
    """
    Calcula la matriu de scores [n_paragraphs x n_images].

    Per cada parell (paràgraf_i, imatge_j) calcula el CLIP score.
    Retorna una matriu com a llista de llistes:
      matrix[i][j] = CLIP score del paràgraf i amb la imatge j
    """
    n_pars  = len(paragraphs)
    n_imgs  = len(images)

    # Aplanem tots els parells (par_i, img_j) en un batch pla
    all_images   = []
    all_captions = []
    pair_index   = []   # (par_idx, img_idx)

    for par_idx, par in enumerate(paragraphs):
        for img_idx, img in enumerate(images):
            all_images.append(img)
            all_captions.append(par)
            pair_index.append((par_idx, img_idx))

    # Inicialitzem la matriu
    matrix = [[0.0] * n_imgs for _ in range(n_pars)]

    # Processar en batches
    for start in range(0, len(all_images), batch_size):
        end     = min(start + batch_size, len(all_images))
        scores  = compute_scores_batch(
            clip_model, clip_processor, device,
            all_images[start:end], all_captions[start:end]
        )
        for (par_idx, img_idx), score in zip(pair_index[start:end], scores):
            matrix[par_idx][img_idx] = round(float(score), 4)

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Processament d'un split
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split_name, clip_model, clip_processor, device,
                  batch_size, save_all=False):
    print(f"\n{'='*60}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")

    split_data = load_dataset(DATASET_PATH, split=split_name)
    n_total    = len(split_data)
    print(f"  Mostres: {n_total}\n")

    mismatches  = []   # mostres on CLIP no identifica la solució
    all_results = []   # tots els resultats (si save_all=True)
    errors      = 0

    # Comptadors
    n_no_caption      = 0
    n_correct         = 0
    n_mismatch        = 0
    clip_scores_sol   = []   # score mitjà de la imatge solució
    clip_scores_pred  = []   # score mitjà de la imatge predita per CLIP

    for sample in tqdm(split_data, desc=f"  {split_name}", total=n_total):
        sid        = sample["sample_id"]
        sol_idx    = sample["solution_index"]
        if sol_idx < 0:
            sol_idx = 0
        options    = sample["options"]
        paragraphs = get_paragraphs(sample)

        if not paragraphs:
            n_no_caption += 1
            continue

        # Carregar totes les imatges de les opcions
        try:
            images = [load_image(opt) for opt in options]
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  ⚠️  Error imatge mostra {sid}: {e}")
            continue

        # Matriu [n_pars x n_imgs]
        try:
            matrix = compute_paragraph_image_matrix(
                clip_model, clip_processor, device,
                images, paragraphs, batch_size
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  ⚠️  Error CLIP mostra {sid}: {e}")
            continue

        # Mitjana dels scores de tots els paràgrafs per cada imatge
        # mean_scores[j] = mitjana de matrix[i][j] per tots els i
        n_imgs       = len(images)
        mean_scores  = [
            round(statistics.mean(matrix[par_idx][img_idx]
                                  for par_idx in range(len(paragraphs))), 4)
            for img_idx in range(n_imgs)
        ]

        # Imatge predita per CLIP (la de score màxim)
        clip_pred_idx   = mean_scores.index(max(mean_scores))
        score_solution  = mean_scores[sol_idx]
        score_predicted = mean_scores[clip_pred_idx]
        is_correct      = (clip_pred_idx == sol_idx)

        if is_correct:
            n_correct += 1
        else:
            n_mismatch += 1

        clip_scores_sol.append(score_solution)
        clip_scores_pred.append(score_predicted)

        # Construïm el registre de la mostra
        record = {
            "sample_id":        sid,
            "split":            split_name,
            "solution_index":   sol_idx,
            "clip_pred_index":  clip_pred_idx,
            "is_correct":       is_correct,
            "n_options":        n_imgs,
            "n_paragraphs":     len(paragraphs),
            "mean_scores":      mean_scores,
            "score_solution":   score_solution,
            "score_predicted":  score_predicted,
            "score_gap":        round(score_predicted - score_solution, 4),
            # Scores per paràgraf de cada opció
            "paragraph_scores_matrix": matrix,
            # Primer paràgraf (per referència)
            "first_paragraph":  paragraphs[0][:300],
        }

        if not is_correct:
            mismatches.append(record)

        if save_all:
            all_results.append(record)

    # ── Estadístiques ─────────────────────────────────────────────────────────
    n_processed = n_correct + n_mismatch
    pct_correct  = 100 * n_correct  / n_processed if n_processed > 0 else 0
    pct_mismatch = 100 * n_mismatch / n_processed if n_processed > 0 else 0

    summary_lines = [
        f"Anàlisi d'alineació CLIP - Split: {split_name}",
        f"Generat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*60}",
        f"",
        f"RESUM GENERAL",
        f"  Mostres totals         : {n_total}",
        f"  Mostres processades    : {n_processed}",
        f"  Sense caption          : {n_no_caption}",
        f"  Errors                 : {errors}",
        f"",
        f"ALINEACIÓ CLIP vs SOLUTION_INDEX",
        f"  ✅ Correctes (CLIP = solució) : {n_correct:6d}  ({pct_correct:.1f}%)",
        f"  ❌ Incorrectes (CLIP ≠ solució): {n_mismatch:6d}  ({pct_mismatch:.1f}%)",
        f"",
        f"CLIP SCORES (mitjana de paràgrafs)",
        f"  Score imatge solució:",
    ]

    if clip_scores_sol:
        summary_lines += [
            f"    Min    : {min(clip_scores_sol):.4f}",
            f"    Max    : {max(clip_scores_sol):.4f}",
            f"    Mitjà  : {statistics.mean(clip_scores_sol):.4f}",
            f"    Mediana: {statistics.median(clip_scores_sol):.4f}",
            f"",
            f"  Score imatge predita per CLIP:",
            f"    Min    : {min(clip_scores_pred):.4f}",
            f"    Max    : {max(clip_scores_pred):.4f}",
            f"    Mitjà  : {statistics.mean(clip_scores_pred):.4f}",
            f"    Mediana: {statistics.median(clip_scores_pred):.4f}",
        ]

    if mismatches:
        gaps = [m["score_gap"] for m in mismatches]
        summary_lines += [
            f"",
            f"  Gap (score_predicció - score_solució) als mismatches:",
            f"    Min    : {min(gaps):.4f}",
            f"    Max    : {max(gaps):.4f}",
            f"    Mitjà  : {statistics.mean(gaps):.4f}",
            f"    Mediana: {statistics.median(gaps):.4f}",
            f"",
            f"  Distribució del gap:",
        ]
        for thr in [0.01, 0.02, 0.05, 0.10, 0.20]:
            n_big = sum(1 for g in gaps if g >= thr)
            summary_lines.append(
                f"    gap >= {thr:.2f}: {n_big:6d} mostres ({100*n_big/len(gaps):.1f}%)"
            )

    summary_lines += [
        f"",
        f"INTERPRETACIÓ",
        f"  Un {pct_mismatch:.1f}% de les mostres tenen una caption que",
        f"  CLIP considera més afí a una imatge diferent de la solució.",
        f"  Això suggereix que el text no sempre descriu la imatge",
        f"  'correcta' del dataset, o que la relació és narrativa/contextual",
        f"  en lloc de purament visual.",
    ]

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    return mismatches, all_results, summary_text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",     nargs="+",
                        default=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--save_all",   action="store_true",
                        help="Guarda també les mostres correctes")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Dispositiu : {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nCarregant CLIP ({CLIP_MODEL})...")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()
    print("  CLIP carregat ✅")

    for split in args.splits:
        mismatches, all_results, summary_text = process_split(
            split, clip_model, clip_processor, device,
            args.batch_size, args.save_all
        )

        # Guardar mismatches
        mismatch_path = os.path.join(args.output_dir, f"mismatches_{split}.json")
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(mismatches, f, ensure_ascii=False, indent=2)
        print(f"\n  ✅ Mismatches guardats: {mismatch_path} ({len(mismatches)} mostres)")

        # Guardar summary
        summary_path = os.path.join(args.output_dir, f"summary_{split}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"  ✅ Resum guardat: {summary_path}")

        # Guardar tots els resultats (opcional)
        if args.save_all and all_results:
            all_path = os.path.join(args.output_dir, f"all_results_{split}.json")
            with open(all_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Tots els resultats guardats: {all_path}")

    print(f"\n{'='*60}")
    print(f"  Anàlisi completada. Resultats a: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()