# Proposta i Millora de l'Arquitectura - Grup 09

L'arquitectura base utilitzada és un model **bi-encoder** que combina **BERT** (per al processament de text) i **ResNet-18** (per a la visió per computador), optimitzat per a la tasca de recuperació de vinyetes.

---

## Millores Implementades

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

## Propostes de Millora Futura (Roadmap)

Basant-nos en l'anàlisi de resultats actual (R@10 de 0.0536), proposem les següents línies de treball per a la Fase 2:

*   **Unfreezing Progressiu**: Actualment els encoders estan parcialment congelats. Proposem "descongelar" les darreres capes de la ResNet-18 per fer un *fine-tuning* específic sobre l'estil visual dels còmics (línies, textures i onomatopeies), que difereix de les imatges reals de ImageNet.
*   **Data Augmentation Específic**: Implementar rotacions lleugeres, canvis de contrast i soroll per augmentar la variabilitat del dataset i millorar la generalització del model.
*   **Hard Negative Mining Avançat**: Ajustar els paràmetres $\alpha$ i $\beta$ de la `MultiSimilarityLoss` per ser encara més agressius amb els exemples que el model confon actualment (vinyetes amb descripcions molt genèriques).