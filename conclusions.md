# Conclusions — Comics Retrieval

## 1. Resum dels resultats

El projecte ha abordat el problema de recuperació text-imatge sobre el dataset COMIC-PAP, implementant i avaluant diverses configuracions de model basades en la combinació de **SentenceBERT** (encoder de text) i **ResNet50** (encoder d'imatge) projectats a un espai d'embedding comú.

Al llarg de 112 runs amb cerca Bayesiana d'hiperparàmetres, s'ha aconseguit un **R@1 = 0.0458** i un **R@10 = 0.2137** com a millors resultats en el conjunt de validació.

---

## 2. Conclusions per component

### 2.1 SentenceBERT com a encoder de text

La substitució de BERT base per **SentenceBERT** suposa una millora qualitativa important: SentenceBERT genera embeddings de frase directament optimitzats per a tasques de similitud semàntica, a diferència del token [CLS] de BERT base que requeria projecció i fine-tuning extensos. Aquesta elecció és coherent amb la naturalesa de la tasca, on el text és una descripció narrativa (frase o paràgraf) i no una paraula aïllada.

### 2.2 Triplet Margin Loss

La **Triplet Margin Loss** ha demostrat ser clarament superior al simple entrenament per similitud sense estructura de pèrdua mètrica. La millora de R@1 del **+71.5%** respecte a la configuració base (0.0267 → 0.0458) confirma que la funció de pèrdua és capaç d'estructurar l'espai d'embedding de forma que les parelles text-imatge correctes s'apropin i les incorrectes es separin.

L'ús de `pytorch-metric-learning` facilita la mineria automàtica de tríades (anchor, positive, negative) dins de cada batch, aprofitant els negatius semi-difícils que aporten el gradient més informatiu.

### 2.3 Descongelació parcial dels encoders

La descongelació de les últimes capes de BERT (layer 5) i ResNet (layer4) **des del primer epoch** ha estat una decisió clau. Mantenir els encoders completament congelats produïa embeddings estàtics: les capes de projecció aprenien a mapejar sempre els mateixos vectors, sense cap capacitat d'adaptació al domini dels còmics. La descongelació parcial permet un fine-tuning del coneixement visual i lingüístic pre-entrenat sense caure en el problema de l'oblit catastròfic.

### 2.4 Filtratge CLIP Score

El filtratge previ de les parelles text-imatge amb baixa coherència semàntica (CLIP Score baix) millora la **qualitat i consistència** de l'entrenament. Tot i que no supera el R@1 màxim de la configuració sense filtratge, sí que eleva la mitjana de R@1 entre runs (+24% de millora en la mitjana) i redueix la variabilitat. Això suggereix que el dataset conté un nombre significatiu de parelles sorolloses que dificulten l'aprenentatge.

### 2.5 Cerca d'hiperparàmetres (Bayesian + Hyperband)

La cerca Bayesiana amb Hyperband Early Stopping ha permès explorar eficientment l'espai d'hiperparàmetres: el learning rate de **5e-5** i el batch size de **256** s'han identificat com els valors òptims. El batch gran és especialment important per a la Triplet Loss, que necessita diversitat de negatius dins de cada batch.

---

## 3. Limitacions identificades

- **Mida del dataset d'entrenament**: s'ha entrenat amb el 20% de les dades per raons de cost computacional. És previsible una millora significativa al usar el dataset complet.
- **Ambigüitat text-imatge**: les descripcions de COMIC-PAP fan referència sovint al context narratiu del panell anterior, no a la imatge directament. Això limita el sostre de rendiment de qualsevol model d'alineació directa text-imatge.
- **Complexitat visual dels còmics**: panells del mateix còmic comparteixen personatges, estil i paleta de color, fent que el retrieval visual sigui intrínsecament difícil.

---

## 4. Línies de millora futures

En base als resultats obtinguts, les millores planificades per a les pròximes iteracions són:

**a) Data Augmentation**
Aplicar transformacions aleatòries sobre les imatges de còmic durant l'entrenament (rotació, variació de color, crop aleatori) per augmentar la variabilitat del training set i reduir l'overfitting als patrons visuals específics del dataset.

**b) Early Stopping amb marge**
Implementar early stopping basat en la mètrica de validació R@1 amb un marge de tolerància (patience), de manera que l'entrenament s'aturi quan el model no millori durant N epochs consecutives. Això evita el sobreentrenament i redueix el cost computacional dels runs llargs.

**c) Augment del llindar de CLIP Score**
Provar valors de llindar de filtratge CLIP més exigents per quedar-nos únicament amb les parelles de major coherència semàntica. El risc és reduir excessivament el dataset; cal trobar l'equilibri òptim.

---

## 5. Conclusió global

El projecte ha demostrat que la combinació de **SentenceBERT + ResNet50 + Triplet Loss + descongelació parcial + filtratge CLIP** és una arquitectura viable per al problema de recuperació text-imatge en còmics, i que cada component aporta una millora mesurable i justificable sobre la configuració anterior. La recerca sistemàtica d'hiperparàmetres ha permès identificar la configuració òptima (lr=5e-5, bs=256) de forma eficient. Les millores futures basades en data augmentation i early stopping amb marge apunten a superar la barrera del R@1 > 0.05 en el conjunt de validació.
