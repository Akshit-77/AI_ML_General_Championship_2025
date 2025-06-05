import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
import timm

class TestImages(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        self.images = [f for f in os.listdir(folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

def setup_data(data_folder, batch_size=64):
    # Get the right transforms for Swin
    config = timm.data.resolve_model_data_config(
        timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    )
    
    train_transform = timm.data.create_transform(**config, is_training=True)
    val_transform = timm.data.create_transform(**config, is_training=False)
    
    # Load dataset
    dataset = datasets.ImageFolder(data_folder, transform=train_transform)
    class_names = dataset.classes
    
    # Split into train/val (90/10)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_data, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Validation data with different transforms
    val_dataset = datasets.ImageFolder(data_folder, transform=val_transform)
    _, val_data = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, class_names, val_transform

def freeze_model(model, layers_to_train):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Always train the final layer
    for param in model.head.parameters():
        param.requires_grad = True
    
    # Get all the transformer blocks
    blocks = []
    for stage in model.layers:
        for block in stage.blocks:
            blocks.append(block)
    
    # Unfreeze from the end
    num_to_unfreeze = min(layers_to_train, len(blocks))
    for i in range(num_to_unfreeze):
        block_idx = len(blocks) - 1 - i
        for param in blocks[block_idx].parameters():
            param.requires_grad = True

def train_epoch(model, train_loader, val_loader, device, epochs=6):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                           lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}: Loss {train_loss/len(train_loader):.3f}, Acc {acc:.1f}%')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return best_acc

def predict_test_images(model, test_folder, class_names, device, transform):
    test_data = TestImages(test_folder, transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)
            
            for i, filename in enumerate(filenames):
                predictions.append({
                    'image': filename,
                    'prediction': class_names[pred_class[i].item()],
                    'confidence': confidence[i].item()
                })
    
    return predictions

def main():
    # Setup
    train_folder = '/kaggle/input/gc2025/AI-ML GC 2025 Dataset/train'
    test_folder = '/kaggle/input/gc2025/AI-ML GC 2025 Dataset/test'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')
    
    # Load data
    train_loader, val_loader, class_names, val_transform = setup_data(train_folder)
    print(f'Found {len(class_names)} bird species')
    
    # Load model
    model = timm.create_model('swin_base_patch4_window7_224', 
                             pretrained=True, 
                             num_classes=len(class_names))
    model = model.to(device)
    
    # Try different amounts of unfreezing
    layer_options = [1, 2, 4, 8, 12]
    best_accuracy = 0
    best_layers = 0
    
    for num_layers in layer_options:
        print(f'\n--- Training with {num_layers} layers unfrozen ---')
        
        freeze_model(model, num_layers)
        
        # Show how many parameters we're training
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Training {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)')
        
        # Train
        accuracy = train_epoch(model, train_loader, val_loader, device)
        
        if accuracy > best_accuracy:
            print(f'New best: {accuracy:.1f}% (was {best_accuracy:.1f}%)')
            best_accuracy = accuracy
            best_layers = num_layers
            torch.save(model.state_dict(), 'final_best_model.pth')
        else:
            print(f'No improvement: {accuracy:.1f}% vs best {best_accuracy:.1f}%')
            print('Stopping here')
            break
    
    print(f'\nBest result: {best_accuracy:.1f}% with {best_layers} layers')
    
    # Load best model and predict
    model.load_state_dict(torch.load('final_best_model.pth'))
    
    if os.path.exists(test_folder):
        print('\nPredicting test images...')
        results = predict_test_images(model, test_folder, class_names, device, val_transform)
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('predictions.csv', index=False)
        
        # Create submission file
        df['label'] = df['prediction'].apply(lambda x: int(x.split('.')[0]))
        df['ID'] = df['image']
        submission = df[['ID', 'label']]
        submission.to_csv('submission.csv', index=False)
        
        print(f'Predicted {len(results)} images')
        print('Files saved: predictions.csv, submission.csv')

if __name__ == "__main__":
    main()