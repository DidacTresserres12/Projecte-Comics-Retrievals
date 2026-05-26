# Conclusions — Comics Retrieval (Grup 09)

## 1. Resum del procés

Al llarg de tres setmanes, hem dissenyat, implementat i optimitzat un sistema de recuperació text-imatge per al dataset COMIC-PAP, passant per diverses arquitectures i resolent problemes tècnics reals que han condicionat tota l'evolució del projecte.

El punt de partida va ser el codi base del Carles (ResNet18 + BERT + CrossEntropyLoss), que inicialment no produïa cap mètrica útil per culpa d'un bug en la càrrega d'imatges (es llegien com a imatges negres). La detecció i correcció d'aquest bug va ser la primera lliçó important: cal verificar sempre les dades d'entrada abans de culpar l'arquitectura.

Des d'aquell punt de partida fins al model final, hem aconseguit:

| Fase | R@1 | R@10 |
|---|---|---|
| Inicial (bug imatges) | ~0.000 | ~0.000 |
| BERT + ResNet18 + MultiSimilarity | 0.0267 | 0.1565 |
| SentenceBERT + ResNet50 + Triplet | 0.0458 | 0.2137 |
| **Model final** (+ weight decay + 40% dades) | **0.0458** | **0.2328** |

---

## 2. Per què aquest és el millor model

El model final (`firm-sweep-24`) combina totes les decisions que, una a una, han demostrat millorar les mètriques:

**SentenceBERT congelat** és la millor opció per al costat de text: els seus embeddings de frase son semànticament rics i estables des del primer moment. Intentar fer fine-tuning de SBERT no va millorar els resultats i afegia inestabilitat a l'entrenament, de manera que mantenir-lo congelat és la decisió correcta.

**ResNet50 amb layer4 descongelada** aporta features visuals de 2048 dimensions amb capacitat d'adaptar-se als patrons visuals específics dels còmics (traços, colors plans, contorns marcats) sense perdre el coneixement preentrenat d'ImageNet. ResNet18 (512 dim) va demostrar ser insuficient quan SBERT estava congelat.

**TripletMarginLoss amb marge 0.4** és adequada per a un dataset visualment homogeni: un marge gran força una separació clara entre parelles correctes i incorrectes fins i tot quan els panells son visualment similars entre si.

**CLIP filtering a 0.29** elimina les parelles sorolloses del training (textos que descriuen el context narratiu, no la imatge) sense reduir excessivament el dataset. Aplicar-lo únicament al train és crucial; aplicar-lo al val, com vam fer per error durant algunes fases, produeix mètriques no representatives.

**Weight decay = 0.1** afegeix regularització L2 que prevé l'overfitting, especialment important quan s'entrena amb un subset del dataset.

---

## 3. Lliçons apreses

**El pipeline val tant com el model**: el bug de les imatges negres i l'error del CLIP en validació van generar setmanes d'anàlisi de resultats incorrectes. Qualsevol canvi en el pipeline d'avaluació ha de ser verificat immediatament amb casos trivials (un model random hauria de donar R@1 ≈ 1/N).

**La velocitat d'iteració és crítica**: treballar amb el 20-40% del dataset va permetre llançar molts més experiments en el mateix temps. Trobar l'equilibri entre qualitat dels resultats i velocitat d'iteració és una habilitat pràctica fonamental en deep learning.

**Cada component té un raonament**: cada canvi d'arquitectura (SentenceBERT, ResNet50, Triplet Loss, CLIP filter) va millorar les mètriques per raons identificables i justificables, no per atzar. Entendre per què funciona cada component permet predir quines millores tindran impacte.

**Les mètriques han de ser fiables**: descobrir que el val set estava contaminat pel filtre CLIP va obligar a repetir experiments que ja donàvem per bons. Definir correctament les mètriques d'avaluació i assegurar-se que son independents del procés d'entrenament és un prerequisit no negociable.

---

## 4. Què es podria millorar

Sabem que el model es pot millorar encara més, però per temps i coneixement considerem que hem assolit un resultat molt satisfactori per al nivell del projecte:

**Entrenar amb el 100% del dataset**: hem treballat amb el 40% per raons de cost computacional. L'execució final de 150 epochs amb el 40% ens donarà una indicació, però és probable que el dataset complet millori significativament R@1 i R@10.

**Data Augmentation**: aplicar transformacions aleatòries a les imatges de còmic durant el training (rotació, color jitter, crop) augmentaria la variabilitat del dataset efectiu i reduiria l'overfitting.

**Explorar ResNet50 vs ViT**: els Vision Transformers han superat ResNet50 en molts benchmarks visuals. Un encoder ViT preentrenat podria capturar millor l'estil específic dels còmics.

**Hard negative mining explícit**: en lloc de confiar en la mineria automàtica de `pytorch-metric-learning`, implementar un miner que prioritzi negatius visuals similars (altres panells del mateix còmic) podria forçar el model a aprendre distincions més fines.

---

## 5. Conclusió global

Aquest projecte ens ha permès recórrer tot el cicle de treball d'un projecte real de deep learning: des de la depuració d'un bug de dades fins a l'optimització sistemàtica d'hiperparàmetres amb WandB Sweeps, passant per decisions d'arquitectura justificades i la gestió d'errors d'avaluació. 

El resultat final, un model amb R@10 = 0.2328 sobre un dataset de recuperació multimodal complex com COMIC-PAP, representa una millora de més del 400% respecte al punt de partida i demostra que la combinació SentenceBERT + ResNet50 + TripletLoss + CLIP filtering és una arquitectura viable i ben fonamentada per a aquest tipus de problema.

Sabem que hi ha marge de millora, però estem convençuts que les decisions que hem pres estan justificades, els experiments estan ben controlats i els resultats son reals i comparables. Hem après molt.
