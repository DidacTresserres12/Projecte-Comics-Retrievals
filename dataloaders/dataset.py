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
        
        # Gestión de imagen
        # Intentamos sacar la imagen del contexto
        if 'context' in item and len(item['context']) > 0:
            image = item['context'][-1].convert('RGB')
        # Si el contexto falla, probamos con 'panel_image' (por si acaso)
        elif 'panel_image' in item:
            image = item['panel_image'].convert('RGB')
        # Si todo falla, creamos una imagen negra de 224x224
        else:
            from PIL import Image
            image = Image.new('RGB', (224, 224), color=0)
        
        # Gestión de texto
        if 'options' in item and 'solution_index' in item:
            caption = item['options'][item['solution_index']]
        else:
            caption = item.get('previous_panel_caption', "")
        caption = str(caption) if caption is not None else ""

        # 3. Transformaciones
        if self.transform:
            image = self.transform(image)
        # 4. Tokenización
        encoding = self.tokenizer(
            caption, # Ahora estamos seguros de que es un string
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