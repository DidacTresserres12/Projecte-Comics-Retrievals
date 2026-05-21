# Anàlisi de Resultats — Comics Retrieval

## 1. Configuració experimental

El projecte aborda el problema de recuperació d'informació multimodal (text-imatge) sobre el dataset **COMIC-PAP**. L'objectiu és que, donada una descripció textual d'un panell de còmic, el model sigui capaç de recuperar la imatge correcta entre tots els panells del conjunt de validació.

La mètrica principal és **Recall@K (R@K)**: proporció de consultes en les quals la imatge correcta apareix entre els K resultats més similars retornats pel model. S'utilitzen R@1 i R@10.

S'han executat un total de **112 runs** distribuïts en diverses configuracions d'arquitectura, amb cerca Bayesiana d'hiperparàmetres (WandB Sweeps) i Hyperband Early Stopping per aturar els runs clarament inferiors.

---

## 2. Evolució de les arquitectures

### 2.1 Arquitectura base: BERT + ResNet18

La primera arquitectura implementada utilitza **BERT** (`bert-base-uncased`) com a encoder de text i **ResNet18** preentrenat com a encoder d'imatge. Els dos encoders es projecten a un espai d'embedding compartit de dimensió 512 mitjançant capes de projecció lineal (768→1024→512 per a text, 512→1024→512 per a imatge).

La funció de pèrdua utilitzada va ser **MultiSimilarityLoss** de la llibreria `pytorch-metric-learning`, amb normalització L2 a la sortida dels encoders. Els millors resultats d'aquesta configuració van ser:

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0267 |
| R@10 màxim | 0.1565 |

Aquests valors es troben clarament per damunt del valor aleatori esperat (R@1 ≈ 1/N), però mostren que el model té dificultats per aprendre la semàntica específica dels còmics, en part perquè BERT i ResNet18 estaven completament congelats i les capes de projecció rebien sempre els mateixos features fixos.

### 2.2 Incorporació de SentenceBERT + ResNet50 + Triplet Loss + descongelació parcial

Aquesta fase introdueix un conjunt de millores simultànies sobre l'arquitectura base: substitució dels encoders (**SentenceBERT** en lloc de BERT, **ResNet50** en lloc de ResNet18), canvi de funció de pèrdua (**Triplet Margin Loss** en lloc de MultiSimilarityLoss) i descongelació parcial dels encoders (SentenceBERT layer 5 i ResNet50 layer4) des del primer epoch d'entrenament. Els millors resultats d'aquesta configuració van ser:

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0458 |
| R@10 màxim | 0.2137 |

Respecte a l'arquitectura base, el R@1 millora un **+71.5%** i el R@10 un **+36.5%**. La descongelació des del primer epoch va resultar clau: mantenir els encoders completament congelats produïa embeddings estàtics que no permetien aprendre cap distinció útil entre parelles. SentenceBERT aporta embeddings de frase semànticament rics des de l'inici, i ResNet50 proporciona features visuals significativament més expressives que ResNet18.

### 2.3 Arquitectura amb filtratge CLIP

La tercera configuració incorpora un **filtre de CLIP Score** previ a l'entrenament: s'eliminen del dataset els exemples amb baixa coherència semàntica text-imatge (parelles redundants o sense contingut informatiu), seleccionant únicament els exemples on la similitud CLIP supera un llindar determinat.

| Mètrica | Valor |
|---------|-------|
| R@1 màxim | 0.0458 |
| R@10 màxim | 0.2099 |

El filtratge CLIP no millora el R@1 màxim, però sí que redueix la variabilitat entre runs i millora la consistència del training (les pèrdues d'entrenament baixen més ràpidament). El millor run d'aquesta configuració (`confused-sweep-1`, batch_size=256, lr=5e-5, 5 epochs) assoleix el R@1 màxim global de tot el conjunt d'experiments.

---

## 3. Anàlisi dels hiperparàmetres

### 3.1 Learning rate

| LR | R@1 mitjà | R@1 màxim |
|----|-----------|-----------|
| 2e-5 | 0.0208 | 0.0305 |
| 5e-5 | 0.0311 | 0.0458 |
| 1e-4 | 0.0271 | 0.0420 |

El learning rate de **5e-5** és el més estable i el que dona millors resultats tant en mitjana com en màxim. Un LR de 2e-5 és massa conservador (el model no adapta prou els encoders descongelats), mentre que 1e-4 pot generar oscil·lació en els pesos de BERT i ResNet.

### 3.2 Batch size

| Batch size | R@1 mitjà | R@1 màxim |
|------------|-----------|-----------|
| 64 | 0.0286 | 0.0420 |
| 128 | 0.0285 | 0.0420 |
| 256 | 0.0276 | 0.0458 |
| 512 | 0.0191 | 0.0191 |

El batch size de **256** és el que dona el millor R@1 màxim. La Triplet Loss es beneficia de batches grans perquè disposa de més negatius per formar tríades informatives. El batch de 512 mostra pitjors resultats probablement per limitacions de memòria de la GPU que van forçar la reducció del gradient acumulat.

### 3.3 Nombre d'epochs

| Epochs | R@1 mitjà | R@1 màxim | R@10 màxim |
|--------|-----------|-----------|------------|
| 5 | 0.0273 | 0.0458 | 0.2099 |
| 10 | 0.0291 | 0.0458 | 0.2137 |

10 epochs millora lleugrement la mitjana, però el màxim és equivalent. Amb Hyperband Early Stopping, els runs de 10 epochs que no progressen s'aturen aviat, pel que el cost computacional és raonable.

---

## 4. Comparació global de configuracions

| Arquitectura | R@1 màxim | R@10 màxim | R@1 mitjà |
|---|---|---|---|
| BERT + ResNet18 + MultiSimilarityLoss (base) | 0.0267 | 0.1565 | 0.0204 |
| SentenceBERT + ResNet50 + Triplet + Unfreeze | 0.0458 | 0.2137 | 0.0285 |
| SentenceBERT + ResNet50 + Triplet + Unfreeze + CLIP filter | **0.0458** | 0.2099 | **0.0353** |

La combinació SentenceBERT + ResNet50 + Triplet Loss + descongelació parcial + filtratge CLIP és la que ofereix la millor combinació de rendiment màxim i consistència entre runs (R@1 mitjà superior).

---

## 5. Discussió

Els valors de R@1 obtinguts (~4.6%) poden semblar baixos en termes absoluts, però cal contextualitzar-los:

- El dataset COMIC-PAP és un problema de retrieval molt exigent: el conjunt de validació conté centenars de panells visalment similars entre si (mateixos personatges, estils, paletes de color).
- L'arquitectura treballa amb textos extrets de descripcions de panells anteriors, que sovint no descriuen directament la imatge de la parella, sinó el context narratiu. Això fa que l'alineació text-imatge sigui inherentment ambigua.
- El model ha estat entrenat amb el **20% del dataset** per raons de cost computacional durant el sweep; és previsible que entrenar amb el dataset complet millori significativament les mètriques.

El salt de millora de la Triplet Loss respecte a la configuració base és el resultat més rellevant: indica que la funció de pèrdua sí que és capaç d'empènyer els embeddings correctes a apropar-se i els incorrectes a separar-se en l'espai d'embedding, fins i tot amb dades complexes com els còmics.

---

## 6. Millors hiperparàmetres identificats

El run amb major R@1 (`confused-sweep-1`) utilitza la següent configuració:

| Hiperparàmetre | Valor òptim |
|---|---|
| Arquitectura | SentenceBERT-ResNet50-Triplet-CLIPfilter |
| batch_size | 256 |
| learning_rate | 5e-5 |
| epochs | 5 |
| embedding_dim | 512 |
| Encoders descongelats | BERT layer 5, ResNet layer4 |

Aquesta configuració s'utilitza com a punt de partida per a les millores futures descrites a `arquitectures.md`.
