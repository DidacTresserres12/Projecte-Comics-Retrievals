# Guia d'Execució i Proves (Test) - Grup 09

Aquest document detalla el protocol per reproduir l'avaluació del model de Retrieval de còmics i el funcionament de la demo personalitzada. Totes les proves s'han realitzat a la VM d'Intel·ligència Artificial de la UAB.

---

## 1. Configuració de l'Entorn

Per garantir que el script `test.py` trobi les llibreries de C++ actualitzades de Conda i no utilitzi les del sistema (evitant errors de `GLIBCXX`), cal executar la següent comanda abans de qualsevol prova:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 2. Avaluació Global del Dataset

L'avaluació calcula la capacitat del model per trobar la vinyeta correcta entre tot el conjunt de test (262 exemples). El script ha estat optimitzat per carregar directament els fitxers de dades en format `.arrow` des de la carpeta del servidor, evitant problemes de quota de disc.

### Execució

```bash
python test.py --model_path trained_models/model_best.pth --eval --device cuda
```

### Mètriques Obtingudes (Resultats Reals)

Després de processar el set de test amb el model final, els resultats oficials del grup són:

- **Recall@1:** `0.045`
- **Recall@10:** `0.19`
- **MRR:** `0.009`

> **Nota tècnica:** Tot i que els valors són baixos a causa de la complexitat del dataset COMIC-PAP, representen un rendiment significativament superior a l'atzar (`~0.001`), validant l'aprenentatge del model.

---

## 3. Demo Personalitzada (Validació Qualitativa)

El mode demo permet introduir una imatge i una llista de frases candidates per observar com el model puntua la similitud del cosinus entre elles.

### Exemple d'execució

```bash
python test.py --model_path trained_models/model_best.pth --image demo_panel.png --texts "un coche de policia" "un bosque" --device cuda
```

### Resultat esperat

En les nostres proves, el model ha demostrat coherència semàntica assignant una puntuació més alta a la descripció correcta:

```text
Frase: "un coche de policia" -> Similitud: 0.3311
Frase: "un bosque" -> Similitud: 0.2342
```

---

## 4. Paràmetres del Script

| Paràmetre       | Tipus     | Descripció |
|----------------|-----------|-------------|
| `--model_path` | `string`  | Obligatori. Ruta al fitxer `.pth` entrenat. |
| `--eval`       | `flag`    | Activa l'avaluació sobre el dataset de test complet. |
| `--image`      | `string`  | [Demo] Ruta a la imatge per a la prova individual. |
| `--texts`      | `string+` | [Demo] Textos candidats separats per espais. |
| `--device`     | `string`  | Dispositiu de càlcul (`cuda` o `cpu`). |

---

## 5. Troubleshooting (Solució de Problemes)

### Disk Quota Exceeded

Si apareix aquest error, neteja la memòria temporal amb:

```bash
rm -rf ~/.cache/huggingface/datasets
```

El script actual ja minimitza aquest risc amb la càrrega directa.

### GLIBCXX_3.4.30 not found

Recorda executar l'`export` del punt 1 per carregar les llibreries de Conda.

### CUDA Out of Memory

Pots reduir el `batch_size` dins del fitxer `test.py` (actualment a `32`).
