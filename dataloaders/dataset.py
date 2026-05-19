import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ComicsPAPDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        # Guardamos el dataset como 'self.ds'
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Transformación para ResNet
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        item = self.ds[idx]

        # Imatge correcta
        sol_idx = item['solution_index']
        if sol_idx < 0:
            sol_idx = 0
        image = item['options'][sol_idx].convert('RGB')

        # Text
        caption = item.get('previous_panel_caption', "") or ""

        # Transformació de la imatge
        image = self.transform(image)

        # Tokenización
        encoding = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }