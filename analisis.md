# Anàlisi de Resultats — Comics Retrieval (Grup 09)

## 1. Configuració experimental

El projecte aborda el problema de recuperació d'informació multimodal (text-imatge) sobre el dataset **COMIC-PAP**. L'objectiu és que, donada una descripció textual d'un panell de còmic, el model sigui capaç de recuperar la imatge correcta entre tots els panells del conjunt de validació.

Les mètriques principals utilitzades son:
- **R@1**: proporció de consultes on la imatge correcta és la primera retornada.
- **R@10**: proporció de consultes on la imatge correcta apareix entre les 10 primeres.
- **MRR (Mean Reciprocal Rank)**: mitjana de 1/rank de la imatge correcta. Captura si el model "quasi encerta" (és sensible a posicions 2 i 3, que R@1 no veu).

S'han executat un total de **~150 runs** distribuïts en diverses configuracions d'arquitectura, amb cerca Bayesiana d'hiperparàmetres (WandB Sweeps) i Hyperband Early Stopping per aturar els runs clarament inferiors.

---

## 2. Evolució cronològica de les arquitectures

### 2.1 Model inicial: ResNet18 + BERT + CrossEntropyLoss (Codi del Carles)

La primera implementació partia del codi base proporcionat pel Carles, que combinava **BERT** (`bert-base-uncased`) com a encoder de text i **ResNet18** com a encoder d'imatge, amb **CrossEntropyLoss** com a funció de pèrdua.

En les primeres execucions, les mètriques R@1 i R@10 sortien sempre a zero o amb valors completament aleatoris. Després d'una anàlisi exhaustiva del pipeline, vam detectar que el problema no era l'arquitectura sinó la **càrrega del dataset**: les imatges s'estaven llegint com a imatges negres (tensors de zeros) perquè la ruta de les imatges en el dataloader era incorrecta. Un cop corregida la importació del dataset i verificat que les imatges es carregaven correctament, el model ja mostrava valors de mètriques no trivials.

Amb el bug de les imatges resolt, vam entendre el flux complet del model del Carles i vam confirmar que la CrossEntropyLoss no era adequada per a una tasca de retrieval (és una pèrdua de classificació, no de similitud).

### 2.2 ResNet18 + BERT + MultiSimilarityLoss + cerca exhaustiva d'hiperparàmetres

Amb el dataset funcionant correctament, vam passar a una primera millora significativa: substituir la CrossEntropyLoss per **MultiSimilarityLoss** de la llibreria `pytorch-metric-learning`. Aquesta funció mina parelles difícils (*hard negatives*) i és molt més adequada per a tasques de retrieval.

Addicionalment, vam implementar **normalització L2** a la sortida dels dos encoders (vectors a la hiperesfera unitària) i vam llançar una cerca exhaustiva d'hiperparàmetres mitjançant WandB Sweeps. Els hiperparàmetres explorats van ser:

| Hiperparàmetre | Valors provats |
|---|---|
| Learning rate | `1e-3`, `5e-4`, `1e-4`, `5e-5`, `2e-5` |
| Batch size | `32`, `64`, `128`, `256` |
| Embedding dim | `128`, `256`, `512`, `1024` |
| Epochs | `5`, `10`, `15`, `20` |
| Margin | `0.1`, `0.2`, `0.3`, `0.5` |
| Freeze encoders | `True`, `False` |

Els millors resultats d'aquesta fase van ser:

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0267 |
| R@10 màxim | 0.1565 |

Un problema que vam identificar en aquesta fase és que el dataset era molt gran i les execucions trigaven molt, cosa que feia que el cicle d'iteració (llançar sweep → esperar resultats → analitzar → ajustar) fos molt lent. Per resoldre-ho vam decidir treballar amb el **20% del dataset** durant les fases de cerca d'hiperparàmetres.

### 2.3 SentenceBERT + ResNet50 + TripletMarginLoss + descongelació parcial

Basant-nos en els resultats anteriors i les recomanacions del professor Carles, vam implementar un conjunt de millores simultànies:

- **BERT → SentenceBERT** (`all-MiniLM-L6-v2`): genera embeddings de frase optimitzats per a similitud semàntica via mean pooling, molt més adequats que el token `[CLS]` de BERT per a descripcions narratives.
- **ResNet18 → ResNet50**: 2048 features de sortida en lloc de 512, molt més expressiu visualment.
- **MultiSimilarityLoss → TripletMarginLoss**: treballa amb tríades (àncora, positiu, negatiu) i imposa explícitament un marge entre distàncies. Amb batches grans, la mineria automàtica de negatius semi-difícils proporciona gradients molt informatius.
- **Descongelació parcial**: es descongelaven SentenceBERT des de la layer 5 i ResNet50 des de layer4, permetent fine-tuning sense oblit catastròfic.

Els millors resultats d'aquesta fase van ser:

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0458 |
| R@10 màxim | 0.2137 |

### 2.4 Filtratge CLIP + anàlisi de qualitat del dataset

Vam implementar un pas de neteja del dataset basat en **CLIP Score**: per a cada parella text-imatge del dataset, vam calcular la similitud semàntica amb un model CLIP preentrenat. Les parelles per sota d'un llindar determinat (`clip_threshold`) s'eliminaven de l'entrenament.

Per decidir el llindar adequat, vam generar gràfiques de distribució dels CLIP scores (fitxer `analyse_captions.py` i `analyse_solution_alignment.py`), que mostraven que una part significativa del dataset tenia scores molt baixos, corresponent a parelles on el text descriu el context narratiu del panell anterior (no el contingut visual de la imatge actual). Vam explorar llindars entre 0.25 i 0.32.

En aquest punt vam cometre un error important: **el filtre CLIP es va aplicar tant al train com al validation set**, cosa que feia que les mètriques de validació estiguessin inflades artificialment (s'avaluava sobre un subconjunt de mostres fàcils, no sobre el dataset real complet). Totes les execucions d'aquesta fase (26T20, 26T20_06, 26T20_08) van tenir mètriques no comparables per aquest motiu.

### 2.5 Millores recomanades pel professor + detecció del bug del CLIP en validació

A partir de les recomanacions del professor Carles, vam implementar diverses millores:

- **ResNet50 → ResNet18**: el professor va argumentar que amb batch size 512 i SBERT congelat, ResNet18 podria convergir més ràpid. Els resultats van empitjorar: amb menys features visuals (512 vs 2048) i SBERT congelat, la projecció tenia massa poca informació per aprendre l'alineació. Vam revertir a ResNet50.
- **Mètrica MRR**: vam afegir el Mean Reciprocal Rank com a mètrica addicional per tenir més informació sobre la posició de la imatge correcta en el rànquing.
- **SBERT completament congelat**: el professor va recomanar congelar SBERT completament i descongelar només ResNet50 layer4.
- **Guardar el millor punt de cada mètrica de forma independent**: vam detectar que `val/best_R@10` s'estava guardant amb el valor del epoch on `R@1` era màxim, no on `R@10` era màxim. Es va corregir perquè cada mètrica es tracki de forma independent.
- **Bug del CLIP en validació**: vam detectar i corregir que el filtre CLIP s'estava aplicant erròniament al validation set, cosa que havia estat inflant les mètriques de validació des de la implementació del CLIP.

Un cop corregit el bug del CLIP en validació (execució 26T20_10), els resultats van ser reals i comparables:

| Sweep | Arquitectura | R@1 màxim | R@10 màxim |
|---|---|---|---|
| 26T20 (CLIP en val — NO VÀLID) | ResNet50 + SBERTcongelat | 0.057 | 0.213 |
| 26T20_06 (CLIP en val — NO VÀLID) | ResNet50 → revert | 0.057 | 0.213 |
| 26T20_08 (CLIP en val — NO VÀLID) | ResNet50 + bs512 | 0.064 | 0.167 |
| **26T20_10 (sense CLIP en val — VÀLID)** | ResNet50 + SBERTcongelat | ~0.045 | ~0.198 |

### 2.6 Exploració ResNet18 vs ResNet50 sense CLIP en validació

Amb les mètriques ja fiables, vam tornar a provar ResNet18 per veure si donava millors resultats (sweep 26T20_10 vs 26T20_11). La comparació va confirmar que **ResNet50 és clarament superior** quan SBERT està congelat: amb ResNet18 els embeddings visuals de 512 dimensions no eren prou expressius per aprendre l'alineació text-imatge en l'espai compartit.

### 2.7 Weight Decay + 40% del dataset — Millor model final (26T19)

L'última millora explorada va ser augmentar el **weight decay** de 0.01 a **0.1** en l'optimitzador AdamW, per afegir una regularització L2 més forta que evités l'overfitting a les dades de training. Simultàniament, vam augmentar el subset d'entrenament del 20% al **40%** per donar al model més dades sense disparar el temps d'execució.

Amb **ResNet50 + SBERT congelat + weight_decay=0.1 + 40% dataset**, el millor run (`firm-sweep-24`) va obtenir:

| Mètrica | Valor |
|---------|-------|
| R@1 | 0.0458 |
| R@10 | **0.2328** |
| MRR | 0.1002 |

Aquest és el millor model obtingut en tot el projecte, i és el que s'utilitza per a l'execució final de 150 epochs.

---

## 3. Anàlisi dels hiperparàmetres (sweep final 26T19)

### 3.1 Learning rate

| LR | R@1 mitjà | R@10 màxim |
|-----|-----------|------------|
| 1e-4 | 0.0458 | 0.2328 |
| 5e-5 | 0.0344 | 0.2137 |
| 2e-5 | 0.0224 | 0.1718 |
| 1e-5 | 0.0182 | 0.1527 |

**1e-4** és el learning rate òptim: prou gran per adaptar ResNet50 layer4 amb els nous gradients, però no tant com per desestabilitzar els pesos preentrenats.

### 3.2 Clip threshold

| Threshold | R@10 màxim |
|-----------|------------|
| 0.29 | **0.2328** |
| 0.27 | 0.2137 |
| 0.30 | 0.1908 |
| 0.0  | 0.1679 |

El threshold de **0.29** és l'òptim: filtra les parelles més sorolloses però conserva prou dades per a un training robust.

### 3.3 Margin

| Margin | R@10 màxim |
|--------|------------|
| 0.4 | **0.2328** |
| 0.3 | 0.1984 |
| 0.2 | 0.2137 |
| 0.1 | 0.1527 |

Un marge més gran (**0.4**) imposa una separació més estricta entre parelles correctes i incorrectes, cosa que beneficia el retrieval en un dataset visualment homogeni com COMIC-PAP.

---

## 4. Comparació global de configuracions

| Fase | Arquitectura | R@1 màxim | R@10 màxim |
|---|---|---|---|
| 2.1 | ResNet18 + BERT + CrossEntropy | ~0.000 | ~0.000 (bug imatges) |
| 2.2 | ResNet18 + BERT + MultiSimilarity | 0.0267 | 0.1565 |
| 2.3 | ResNet50 + SentenceBERT + Triplet | 0.0458 | 0.2137 |
| 2.4–2.5 | + CLIP filter (val contaminat) | NO VÀLID | NO VÀLID |
| 2.6 | ResNet50 + SBERT congelat (val net) | 0.0458 | 0.1984 |
| **2.7** | **+ weight_decay=0.1 + 40% dataset** | **0.0458** | **0.2328** |

---

## 5. Discussió

Els valors de R@1 (~4.6%) cal contextualitzar-los: el dataset COMIC-PAP conté panells visualment molt similars (mateixos personatges, estil, paleta), i els textos sovint descriuen el context narratiu del panell anterior en lloc del contingut visual directe. Això crea una ambigüitat intrínseca que limita el sostre de rendiment de qualsevol model d'alineació directa text-imatge.

El R@10 de **0.2328** és el resultat més rellevant: indica que en quasi un quart de les consultes, la imatge correcta apareix entre les 10 primeres d'un ranking de centenars de panells. Tenint en compte la dificultat del problema i que s'ha entrenat amb el 40% de les dades, és un resultat sòlid.
