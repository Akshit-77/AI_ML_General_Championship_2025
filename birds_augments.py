import os
import random
import shutil
from PIL import Image
from torchvision import transforms
import torch

def augment_images(input_folder, output_folder):
    """
    Create 4x more images by applying different transformations
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Simple transformations
    flip_rotate = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(30)
    ])
    
    move_vertical = transforms.RandomAffine(degrees=0, translate=(0, 0.2))
    
    flip_move = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0))
    ])
    
    # Get all bird folders
    bird_folders = [f for f in os.listdir(input_folder) 
                   if os.path.isdir(os.path.join(input_folder, f))]
    
    print(f"Found {len(bird_folders)} bird types")
    
    for bird_type in bird_folders:
        print(f"Working on {bird_type}...")
        
        input_path = os.path.join(input_folder, bird_type)
        output_path = os.path.join(output_folder, bird_type)
        os.makedirs(output_path, exist_ok=True)
        
        # Get all images
        images = [f for f in os.listdir(input_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_file = os.path.join(input_path, img_name)
            
            try:
                img = Image.open(img_file).convert('RGB')
                base_name = img_name.split('.')[0]
                ext = '.' + img_name.split('.')[-1]
                
                # Save 4 versions
                img.save(os.path.join(output_path, f"{base_name}_orig{ext}"))
                
                img2 = flip_rotate(img)
                img2.save(os.path.join(output_path, f"{base_name}_flip{ext}"))
                
                img3 = move_vertical(img)
                img3.save(os.path.join(output_path, f"{base_name}_move{ext}"))
                
                img4 = flip_move(img)
                img4.save(os.path.join(output_path, f"{base_name}_combo{ext}"))
                
            except Exception as e:
                print(f"Problem with {img_name}: {e}")
    
    print("Done creating augmented images!")

def smart_augment(train_folder, test_folder, output_folder, multiply_by=2):
    """
    Augment training data while keeping the train/test ratio balanced
    """
    # Count original images
    train_count = 0
    for bird_type in os.listdir(train_folder):
        bird_path = os.path.join(train_folder, bird_type)
        if os.path.isdir(bird_path):
            train_count += len([f for f in os.listdir(bird_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    test_count = len([f for f in os.listdir(test_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Original: {train_count} train, {test_count} test")
    
    # Figure out how much to augment
    total_original = train_count + test_count
    train_ratio = train_count / total_original
    
    target_total = multiply_by * total_original
    target_train = target_total * train_ratio
    augment_factor = target_train / train_count
    
    print(f"Will create {augment_factor:.1f}x training images")
    
    # Create augmented dataset
    os.makedirs(output_folder, exist_ok=True)
    
    # Different ways to change images
    transforms_to_use = [
        transforms.Lambda(lambda x: x),  # original
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(15)
        ]),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ]
    
    how_many_versions = int(augment_factor)
    
    for bird_type in os.listdir(train_folder):
        bird_input = os.path.join(train_folder, bird_type)
        if not os.path.isdir(bird_input):
            continue
            
        bird_output = os.path.join(output_folder, bird_type)
        os.makedirs(bird_output, exist_ok=True)
        
        images = [f for f in os.listdir(bird_input) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(bird_input, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                base_name = img_name.split('.')[0]
                ext = '.' + img_name.split('.')[-1]
                
                # Apply transformations
                for i in range(min(how_many_versions, len(transforms_to_use))):
                    if i == 0:
                        new_name = f"{base_name}_orig{ext}"
                    else:
                        new_name = f"{base_name}_aug{i}{ext}"
                    
                    transformed = transforms_to_use[i](img)
                    transformed.save(os.path.join(bird_output, new_name))
                    
            except Exception as e:
                print(f"Error with {img_name}: {e}")
    
    print("Smart augmentation finished!")

def make_zip(folder_path, zip_name="augmented_birds"):
    """Make a zip file of the folder"""
    print("Creating zip file...")
    shutil.make_archive(zip_name, 'zip', folder_path)
    
    size_gb = os.path.getsize(f"{zip_name}.zip") / (1024**3)
    print(f"Zip created: {zip_name}.zip ({size_gb:.1f} GB)")

# Main execution
if __name__ == "__main__":
    # Set seed for consistent results
    random.seed(42)
    torch.manual_seed(42)
    
    # Simple 4x augmentation
    print("=== Simple 4x Augmentation ===")
    input_dir = "/kaggle/input/birds200/AI-ML GC 2025 Dataset/train"
    output_dir = "/kaggle/working/birds_4x/train"
    
    augment_images(input_dir, output_dir)
    make_zip(output_dir, "birds_4x_augmented")
    
    # Smart balanced augmentation
    print("\n=== Smart Balanced Augmentation ===")
    train_dir = "/kaggle/input/birds200/AI-ML GC 2025 Dataset/train"
    test_dir = "/kaggle/input/birds200/AI-ML GC 2025 Dataset/test"
    smart_output = "/kaggle/working/birds_balanced/train"
    
    smart_augment(train_dir, test_dir, smart_output, multiply_by=2)
    make_zip(smart_output, "birds_balanced_augmented")
    
    print("\nAll done!")