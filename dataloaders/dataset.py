import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ComicsPAPDataset(Dataset):
    """
    Dataset Comics-PAP per a cross-modal retrieval text-imatge.

    Estructura de cada mostra:
      - 'options':               llista de 4 imatges candidates
      - 'solution_index':        índex (0-3) de la imatge correcta
      - 'previous_panel_caption': text llarg (pot tenir salts de línia \\n)

    Retorna per cada mostra:
      - 'image':          la imatge correcta (positiu per a TripletLoss)
      - 'input_ids':      text tokenitzat
      - 'attention_mask': màscara de padding

    Els negatius per a TripletLoss es generen automàticament a train_batch
    a partir dels altres exemples del batch (in-batch negatives).
    """

    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        sol_idx = item['solution_index']
        if sol_idx < 0:
            sol_idx = 0
        image = self.transform(item['options'][sol_idx].convert('RGB'))

        caption = item.get('previous_panel_caption', '') or ''

        encoding = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'image':          image,
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
