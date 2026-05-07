import wandb
import torch

def test(model, test_loader, device="cuda", save:bool= True):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        print(len(images))
        torch.onnx.export(model,  
                          images,  
                          "model.onnx",  #on guardem
                          export_params=True, 
                          opset_version=10,  
                          do_constant_folding=True,  
                          input_names=['input'],  
                          output_names=['output'],  
                          dynamic_axes={'input': {0: 'batch_size'},  
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")