import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# Per què SentenceBERT en lloc de BERT base?
#
# BERT base amb el token [CLS] va ser dissenyat per a tasques de classificació
# de seqüències (p. ex. GLUE), NO per a generar embeddings de frases comparables
# per similitud cosinus. Quan s'usa directament per a retrieval, el token [CLS]
# sol produir representacions anisotròpiques (molts vectors apunten en la
# mateixa direcció) que degraden la qualitat de la similitud cosinus.
#
# SentenceBERT (SBERT) és la mateixa arquitectura Transformer, però pre-entrenada
# específicament amb una xarxa siamesa i Triplet/NLI loss perquè els embeddings
# de frases siguin directament comparables per cosinus. A més, usa mean pooling
# (promig de tots els tokens) en lloc del [CLS] sol, el que dona representacions
# més riques i estables.
#
# Usem 'all-MiniLM-L6-v2':
#   - 6 capes (vs 12 del BERT base) → ×2 més ràpid, menys memòria GPU
#   - Output dim: 384 (vs 768 de BERT base)
#   - Entrenat sobre 1B+ parelles de frases; molt bo per a similitud semàntica
#   - Alternativa de més qualitat: 'all-mpnet-base-v2' (dim=768, més lent)
# ─────────────────────────────────────────────────────────────────────────────

from transformers import AutoModel, AutoTokenizer


class Baseline(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        freeze_sbert=True,
        freeze_resnet=True,
    ):
        """
        Model de cross-modal retrieval text-imatge amb SentenceBERT i ResNet50.

        Canvis respecte a la versió anterior amb BERT base:
        - Substituïm BertModel + token [CLS] per SentenceBERT + mean pooling
        - Mean pooling: promig ponderat pels tokens no-padding (attention_mask)
          → representació més rica que un sol token [CLS]
        - sbert_output_dim = 384 (MiniLM) o 768 (mpnet) en lloc de 768 (BERT base)

        Args:
            embedding_dim: Dimensió de l'espai d'embedding compartit
            sbert_model:   Nom del model SentenceBERT a HuggingFace
            freeze_sbert:  Si True, congela els paràmetres de SBERT
            freeze_resnet: Si True, congela els paràmetres de ResNet
        """
        super(Baseline, self).__init__()

        # ── Encoder de text: SentenceBERT ──
        # Carreguem el transformer subjacent de SBERT (sense la capa pooling)
        # per poder controlar el fine-tuning capa per capa igual que fèiem amb BERT.
        self.sbert = AutoModel.from_pretrained(sbert_model)
        self.sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_model)

        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False

        # ── Encoder d'imatge: ResNet18 ──
        # Canvi ResNet50 → ResNet18: menys paràmetres (11M vs 25M), feature map
        # de 512 en lloc de 2048. Amb SBERT completament congelat, la projecció
        # d'imatge és l'únic lloc on el model pot aprendre; ResNet18 + capes de
        # projecció entrenables pot ser suficient i convergeix més ràpid.
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Dimensions de sortida
        self.sbert_output_dim = self.sbert.config.hidden_size  # 384 per MiniLM, 768 per mpnet
        self.resnet_output_dim = 512  # ResNet18 dona 512 (vs 2048 de ResNet50)

        # ── Capes de projecció a l'espai d'embedding compartit ──
        # Sempre entrenables (no congelades), perquè és aquí on el model aprèn
        # a alinear representacions de text i imatge.
        self.text_projection = nn.Sequential(
            nn.Linear(self.sbert_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(self.resnet_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

        # Preprocessat d'imatges per a ResNet
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # Mean Pooling
    # ─────────────────────────────────────────────────────────────────────────
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling: promig de tots els token embeddings ponderat per la
        màscara d'atenció (ignorem els tokens de padding).

        Per què mean pooling i no [CLS]?
        - El token [CLS] de BERT base codifica informació global però és un
          sol vector que no ha estat entrenat explícitament per a similitud.
        - Mean pooling agrega TOTS els tokens significatius → representació
          més completa i estable, especialment per a textos llargs com les
          captions de còmics (múltiples línies separades per \n).

        Args:
            model_output:   sortida del transformer (last_hidden_state, ...)
            attention_mask: màscara [batch, seq_len], 1=token real, 0=padding

        Returns:
            Tensor [batch, hidden_size] amb el promig ponderat
        """
        token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden]
        # Expandim la màscara per poder multiplicar element a element
        input_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        # Suma dels embeddings dels tokens reals
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
        # Número de tokens reals per mostra (mínim 1 per evitar divisió per zero)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessat
    # ─────────────────────────────────────────────────────────────────────────
    def preprocess_text(self, text_samples):
        encoded = self.sbert_tokenizer(
            text_samples,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return encoded

    def preprocess_image(self, image_paths):
        images = []
        for img in image_paths:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if isinstance(img, Image.Image):
                img = self.image_transform(img)
            images.append(img)
        return torch.stack(images)

    # ─────────────────────────────────────────────────────────────────────────
    # Encoders
    # ─────────────────────────────────────────────────────────────────────────
    def encode_text(self, text_features=None, raw_text=None):
        """
        Codifica text a l'espai d'embedding compartit.

        Flux:
          text → SentenceBERT → mean pooling → text_projection → L2 normalize

        Args:
            text_features: dict amb 'input_ids' i 'attention_mask' (opcional)
            raw_text:      llista de strings (opcional)

        Returns:
            Tensor [batch, embedding_dim], normalitzat L2
        """
        if raw_text is not None:
            text_features = self.preprocess_text(raw_text).to(
                next(self.parameters()).device
            )

        sbert_frozen = not any(p.requires_grad for p in self.sbert.parameters())
        if sbert_frozen:
            with torch.no_grad():
                outputs = self.sbert(**text_features)
                # Mean pooling (en lloc del [CLS] de la versió anterior)
                sentence_emb = self._mean_pooling(outputs, text_features["attention_mask"]).detach()
        else:
            outputs = self.sbert(**text_features)
            sentence_emb = self._mean_pooling(outputs, text_features["attention_mask"])

        # Projecció i normalització L2
        text_embeddings = self.text_projection(sentence_emb)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        return text_embeddings

    def encode_image(self, image_features=None, image_paths=None):
        """
        Codifica imatges a l'espai d'embedding compartit.

        Flux:
          imatge → ResNet50 (sense FC) → aplanat → image_projection → L2 normalize

        Args:
            image_features: tensor d'imatges preprocessades (opcional)
            image_paths:    llista de paths o PIL Images (opcional)

        Returns:
            Tensor [batch, embedding_dim], normalitzat L2
        """
        if image_paths is not None:
            image_features = self.preprocess_image(image_paths)

        resnet_frozen = not any(p.requires_grad for p in self.resnet.parameters())
        if resnet_frozen:
            with torch.no_grad():
                feats = self.resnet(image_features)
                feats = feats.flatten(start_dim=1).detach()
        else:
            feats = self.resnet(image_features)
            feats = feats.flatten(start_dim=1)

        image_embeddings = self.image_projection(feats)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        return image_embeddings

    def forward(self, text_features=None, image_features=None,
                raw_text=None, image_paths=None):
        text_embeddings = None
        image_embeddings = None

        if text_features is not None or raw_text is not None:
            text_embeddings = self.encode_text(text_features, raw_text)

        if image_features is not None or image_paths is not None:
            image_embeddings = self.encode_image(image_features, image_paths)

        return text_embeddings, image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """
        Similitud cosinus entre tots els parells text-imatge.
        Com que els embeddings estan normalitzats L2, el producte escalar
        és equivalent a la similitud cosinus.

        Returns:
            Tensor [batch_text, batch_image]
        """
        return torch.mm(text_embeddings, image_embeddings.t())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Baseline(embedding_dim=512)
    model.to(device)
    model.eval()

    texts = [
        "A superhero fights a villain in the city",
        "Two characters are having a conversation",
        "An explosion destroys a building",
    ]
    images = torch.randn(3, 3, 224, 224).to(device)

    with torch.no_grad():
        text_emb, _ = model(raw_text=texts)
        _, img_emb  = model(image_features=images)

    sim = model.compute_similarity(text_emb, img_emb)
    print("Similarity matrix (Text x Images):")
    print(sim.cpu().numpy())
    print(f"\nText embedding dim: {text_emb.shape}")
    print(f"Image embedding dim: {img_emb.shape}")
