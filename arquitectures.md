# Proposta i Millora de l'Arquitectura - Grup 09

L'arquitectura base utilitzada és un model **bi-encoder** que combina **BERT** (per al processament de text) i **ResNet-18** (per a la visió per computador), optimitzat per a la tasca de recuperació de vinyetes.

---

## Millores Implementades (Fase 1)

La millora més significativa respecte a la implementació inicial ha estat el canvi en l'estratègia d'aprenentatge i la gestió de l'espai latent:

### 1. Pas a Metric Learning
Hem substituït la funció `CrossEntropyLoss` (pròpia de classificació) per **`MultiSimilarityLoss`** de la llibreria *PyTorch Metric Learning*. Aquesta decisió es fonamenta en:
*   **Mineria d'exemples**: La funció se centra a minar parelles "difícils" (*hard negatives*), forçant una separació més clara en l'espai d'embeddings.
*   **Robustesa**: És molt més eficaç per a tasques de *retrieval* on l'objectiu no és predir una classe, sinó ordenar resultats per similitud.

### 2. Normalització d'Embeddings
S'ha implementat una **normalització L2** a la sortida dels dos encoders.
*   Això assegura que tots els vectors resultants es trobin en una hiperesfera d'unitat.
*   Facilita la convergència del model i permet que la similitud del cosinus sigui una mètrica directa i interpretable per al rànquing.

### 3. Exploració d'Hiperparàmetres (Sweep)
Mitjançant una exploració sistemàtica amb *Weights & Biases*, s'han testejat diferents mides de batch i ritmes d'aprenentatge (*Learning Rate*).
*   **Batch Size**: 64 (determinat com el més estable per a la GPU de la VM).
*   **Learning Rate**: 1e-4.
*   **Configuració final**: Aquesta combinació ofereix el millor equilibri entre velocitat de convergència i qualitat del rànquing (**R@10**).

---

## Millores Implementades (Fase 2)

Basant-nos en els resultats de la Fase 1 i les propostes de millora plantejades, aquesta setmana hem implementat les següents modificacions al model i al pipeline d'entrenament:

### 1. Substitució de BERT per SentenceBERT
S'ha substituït **BERT base** (`bert-base-uncased`) per **SentenceBERT** (`sentence-transformers/all-MiniLM-L6-v2`) com a encoder de text. A diferència de BERT, SentenceBERT genera embeddings de frase directament optimitzats per a tasques de similitud semàntica mitjançant *mean pooling* sobre tots els tokens, en comptes de dependre únicament del token `[CLS]`. Això és especialment adequat per a les descripcions narratives de COMIC-PAP, que solen ser frases o paràgrafs complets.

### 2. Substitució de ResNet-18 per ResNet-50
S'ha actualitzat l'encoder visual de **ResNet-18** a **ResNet-50**, que disposa d'una capacitat representacional notablement superior (2048 dimensions de sortida enfront de les 512 de ResNet-18). Ambdós models s'inicialitzen amb pesos preentrenats d'ImageNet i la capa de classificació final s'elimina per obtenir els features visuals purs.

### 3. Substitució de MultiSimilarityLoss per Triplet Margin Loss
S'ha canviat la funció de pèrdua de `MultiSimilarityLoss` a **`TripletMarginLoss`** de la llibreria *PyTorch Metric Learning*. La Triplet Loss treballa amb tríades (àncora, positiu, negatiu) i imposa explícitament que la distància entre l'àncora i el positiu sigui inferior a la distància amb el negatiu per un marge $m$:

$$d(\text{àncora}, \text{positiu}) + m < d(\text{àncora}, \text{negatiu})$$

La mineria automàtica de negatius semi-difícils (*semi-hard negatives*) dins de cada batch proporciona el gradient més informatiu, evitant tant els negatius trivials com els excessivament difícils.

### 4. Descongelació Parcial dels Encoders (Partial Unfreeze)
Implementant la millora proposada a la Fase 1, s'han descongelat les últimes capes dels encoders **des del primer epoch** d'entrenament:
*   **SentenceBERT**: descongelació a partir de la **layer 5** (les capes 0–4 resten congelades).
*   **ResNet-50**: descongelació del bloc **layer4** (l'últim bloc residual).

Mantenir els encoders completament congelats produïa embeddings estàtics que impedien qualsevol adaptació al domini dels còmics. La descongelació parcial permet fine-tuning del coneixement pre-entrenat sense caure en l'oblit catastròfic (*catastrophic forgetting*).

### 5. Filtratge per CLIP Score
S'ha implementat un pas de **neteja del dataset** previ a l'entrenament basat en el CLIP Score: es calcula la similitud semàntica text-imatge per a cada parella del dataset mitjançant un model CLIP i s'eliminen les parelles amb puntuació per sota d'un llindar determinat. Això permet descartar les parelles sorolloses (textos que descriuen el context narratiu i no la imatge directament) que dificulten l'aprenentatge de l'alineació text-imatge.

### 6. Cerca Bayesiana d'Hiperparàmetres amb Hyperband Early Stopping
S'ha substituït la cerca en graella per una **cerca Bayesiana** (`method: "bayes"`) amb WandB Sweeps, que aprèn quines zones de l'espai d'hiperparàmetres donen millors resultats i dirigeix la cerca de forma intel·ligent. S'ha afegit **Hyperband Early Stopping** per aturar automàticament els runs que clarament no progressen en les primeres epochs, estalviant recursos de còmput.

Hiperparàmetres explorats: `learning_rate ∈ {1e-4, 5e-5, 2e-5}`, `batch_size ∈ {64, 128, 256, 512}`, `margin ∈ {0.1, 0.2, 0.3}`, `epochs ∈ {5, 10}`.

---

## Propostes de Millora Futura (Roadmap Fase 3)

Basant-nos en l'anàlisi de resultats actual, proposem les següents línies de treball per a la propera iteració:

*   **Data Augmentation**: Implementar transformacions aleatòries sobre les imatges de còmic durant l'entrenament (`RandomHorizontalFlip`, `ColorJitter`, `RandomCrop`) per augmentar la variabilitat del dataset i reduir l'overfitting als patrons visuals específics del corpus.

*   **Early Stopping amb marge**: Aturar l'entrenament quan el R@1 de validació no millori durant `patience=3` epochs consecutives, amb un marge de tolerància (`min_delta=0.001`) per evitar aturades prematures per oscil·lació. El model guardat serà sempre el del millor epoch.

*   **Augment del llindar de CLIP Score**: Explorar valors de llindar de filtratge CLIP més exigents per retenir únicament les parelles de major coherència semàntica, i estudiar el trade-off entre mida del dataset resultant i qualitat de l'entrenament.
