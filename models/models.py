import torch.nn as nn

# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
#Carles
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms


class Baseline(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        bert_model="bert-base-uncased",
        freeze_bert=True,
        freeze_resnet=True,
    ):
        """
        Baseline model for text-image cross-modal retrieval using BERT and ResNet

        Args:
            embedding_dim: Dimension of the joint embedding space
            bert_model: BERT model version to use
            freeze_bert: Whether to freeze BERT parameters
            freeze_resnet: Whether to freeze ResNet parameters
        """
        super(Baseline, self).__init__()

        # Initialize BERT for text encoding
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Initialize ResNet for image encoding (remove final classification layer)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1]
        )  # Remove final FC layer

        # Freeze ResNet parameters if specified
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Get dimensions of BERT and ResNet outputs
        self.bert_output_dim = self.bert.config.hidden_size  # Usually 768
        self.resnet_output_dim = 2048  # ResNet50 feature dim

        # Projection layers to common embedding space
        self.text_projection = nn.Sequential(
            nn.Linear(self.bert_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(self.resnet_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )

        # Image preprocessing
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_text(self, text_samples):
        """
        Preprocess text using BERT tokenizer

        Args:
            text_samples: List of text strings

        Returns:
            Dictionary of input_ids, attention_mask, etc.
        """
        encoded_text = self.bert_tokenizer(
            text_samples,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return encoded_text

    def preprocess_image(self, image_paths):
        """
        Preprocess images for ResNet

        Args:
            image_paths: List of paths to images or PIL Image objects

        Returns:
            Tensor of preprocessed images
        """
        images = []
        for img in image_paths:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if isinstance(img, Image.Image):
                img = self.image_transform(img)
            images.append(img)
        return torch.stack(images)

    def encode_text(self, text_features=None, raw_text=None):
        """
        Encode text into the joint embedding space

        Args:
            text_features: Preprocessed text features for BERT (optional)
            raw_text: List of raw text strings (optional)

        Returns:
            text_embeddings: Tensor of shape [batch_size, embedding_dim]
        """
        if raw_text is not None:
            text_features = self.preprocess_text(raw_text).to(next(self.parameters()).device)

        # Get BERT embeddings ([CLS] token).
        # Apliquem no_grad NOMÉS als paràmetres de BERT quan està congelat,
        # però sempre fora del bloc de projecció perquè text_projection
        # necessita gradients durant tot l'entrenament.
        bert_frozen = not any(p.requires_grad for p in self.bert.parameters())
        if bert_frozen:
            with torch.no_grad():
                outputs = self.bert(**text_features)
                cls_token = outputs.last_hidden_state[:, 0, :].detach()
        else:
            outputs = self.bert(**text_features)
            cls_token = outputs.last_hidden_state[:, 0, :]

        # Project to embedding space (sempre amb gradients)
        text_embeddings = self.text_projection(cls_token)

        # L2 normalize
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        return text_embeddings

    def encode_image(self, image_features=None, image_paths=None):
        """
        Encode images into the joint embedding space

        Args:
            image_features: Preprocessed image features (optional)
            image_paths: List of image paths or PIL Images (optional)

        Returns:
            image_embeddings: Tensor of shape [batch_size, embedding_dim]
        """
        if image_paths is not None:
            image_features = self.preprocess_image(image_paths)

        # Get ResNet features.
        # Apliquem no_grad NOMÉS quan ResNet està congelat, i sempre fora
        # del bloc de projecció perquè image_projection necessita gradients.
        resnet_frozen = not any(p.requires_grad for p in self.resnet.parameters())
        if resnet_frozen:
            with torch.no_grad():
                feats = self.resnet(image_features)
                feats = feats.flatten(start_dim=1).detach()
        else:
            feats = self.resnet(image_features)
            feats = feats.flatten(start_dim=1)

        # Project to embedding space (sempre amb gradients)
        image_embeddings = self.image_projection(feats)

        # L2 normalize
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        return image_embeddings

    def forward(
        self, text_features=None, image_features=None, raw_text=None, image_paths=None
    ):
        """
        Forward pass through the model

        Args:
            text_features: Preprocessed text features (optional)
            image_features: Preprocessed image features (optional)
            raw_text: List of raw text strings (optional)
            image_paths: List of image paths or PIL Images (optional)

        Returns:
            text_embeddings: Encoded text (if text input provided)
            image_embeddings: Encoded images (if image input provided)
        """
        text_embeddings = None
        image_embeddings = None

        if text_features is not None or raw_text is not None:
            text_embeddings = self.encode_text(text_features, raw_text)

        if image_features is not None or image_paths is not None:
            image_embeddings = self.encode_image(image_features, image_paths)

        return text_embeddings, image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """
        Compute similarity scores between text and image embeddings

        Args:
            text_embeddings: Tensor of shape [batch_size_text, embedding_dim]
            image_embeddings: Tensor of shape [batch_size_image, embedding_dim]

        Returns:
            similarity: Tensor of shape [batch_size_text, batch_size_image]
        """
        similarity = torch.mm(text_embeddings, image_embeddings.t())
        return similarity


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = Baseline(embedding_dim=512)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Example texts
    texts = [
        "A dog playing in the park",
        "A cat sleeping on the couch",
        "A person riding a bicycle",
        "A sunset over the mountains",
        "A plate of delicious food",
    ]

    images = torch.randn(5, 3, 224, 224).to(device)
    # Process text and images
    print("Encoding text and images...")
    with torch.no_grad():
        text_embeddings, _ = model(raw_text=texts)
        _, image_embeddings = model(image_paths=images)

    similarity = model.compute_similarity(text_embeddings, image_embeddings)

    print("\nSimilarity Matrix (Text × Images):")
    print(similarity.cpu().numpy())