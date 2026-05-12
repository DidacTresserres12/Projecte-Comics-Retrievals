# Test Script — Comics Retrieval

Aquest document explica com utilitzar el script `test.py` per avaluar el model entrenat i fer demostracions amb exemples individuals.

---

## Requisits

- Tenir un model entrenat guardat en un fitxer `.pth` (generat per `main.py` a la carpeta `trained_models/`)
- Tenir les dependències instal·lades: `torch`, `transformers`, `torchvision`, `datasets`, `wandb`, `Pillow`

---

## Modes d'ús

El script té dos modes que es poden combinar:

### 1. Mode Demo (exemple individual)

Donada una imatge i una llista de textos candidats, el model calcula la similitud de cada text amb la imatge i indica quin és el més similar.

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --image path/a/la/imatge.jpg \
  --texts "A hero fights a villain" "A dog sleeps on the sofa" "Two characters talk"
```

**Sortida esperada:**
```
Imatge: path/a/la/imatge.jpg
──────────────────────────────────────────────────
  [0.8342]  A hero fights a villain
  [0.2103]  A dog sleeps on the sofa
  [0.4871]  Two characters talk
──────────────────────────────────────────────────
  ✓ Millor coincidència: "A hero fights a villain" (sim=0.8342)
```

---

### 2. Mode Avaluació completa

Avalua el model sobre el dataset de validació complet i calcula les mètriques estàndard de retrieval: **R@1**, **R@5** i **R@10**.

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --eval
```

Per enviar els resultats a Weights & Biases afegeix `--wandb`:

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --eval \
  --wandb
```

**Sortida esperada:**
```
========================================
  Test R@1  : 0.3521
  Test R@5  : 0.6104
  Test R@10 : 0.7238
========================================
```

---

### 3. Ambdós modes alhora

```bash
python test.py \
  --model_path trained_models/model_best.pth \
  --eval \
  --image panel.jpg \
  --texts "A hero fights" "A dog sleeps"
```

---

## Paràmetres

| Paràmetre | Tipus | Per defecte | Descripció |
|---|---|---|---|
| `--model_path` | string | **obligatori** | Ruta al fitxer `.pth` del model entrenat |
| `--embedding_dim` | int | 512 | Dimensió de l'embedding (ha de coincidir amb l'entrenament) |
| `--device` | string | cpu | `cpu` o `cuda` |
| `--image` | string | — | [Demo] Ruta a la imatge d'exemple |
| `--texts` | string+ | — | [Demo] Textos candidats separats per espai |
| `--eval` | flag | — | [Eval] Avalua sobre el dataset de test complet |
| `--wandb` | flag | — | [Eval] Envia els resultats a Weights & Biases |

---

## Mètriques

- **R@K (Recall at K):** proporció de consultes on la resposta correcta apareix entre els K resultats més similars. Com més alt millor (màxim = 1.0).
  - R@1: la resposta correcta és la primera
  - R@5: la resposta correcta és entre les 5 primeres
  - R@10: la resposta correcta és entre les 10 primeres