import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *

def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def make(config, device="cuda"):
    # Carregar Dataset
    raw_dataset = load_dataset("VLR-CVC/ComicsPAP", "caption_relevance")
    
    # Instanciar Model per obtenir el tokenizer
    model = Baseline().to(device)
    tokenizer = model.bert_tokenizer

    # Crear Datasets i Loaders
    train_data = ComicsPAPDataset(raw_dataset['train'], tokenizer)
    test_data = ComicsPAPDataset(raw_dataset['validation'], tokenizer)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # 4. Optimizador parametritzat des del config de pes i biaix
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    return model, train_loader, test_loader, criterion, optimizer