import os
import shutil
import random

# Path where your downloaded dataset folders are located
source_dir = r"C:\Users\KIIT0001\Downloads\archive (1)" 

# Destination path in your project
base_dest = "dataset"
train_dest = os.path.join(base_dest, "Training")
test_dest = os.path.join(base_dest, "Testing")

os.makedirs(train_dest, exist_ok=True)
os.makedirs(test_dest, exist_ok=True)

# Split ratio
split_ratio = 0.8  # 80% training, 20% testing

for category in os.listdir(source_dir):
    cat_path = os.path.join(source_dir, category)
    
    if os.path.isdir(cat_path):
        images = os.listdir(cat_path)
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if len(images) == 0:
            continue
            
        random.shuffle(images)
        
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        train_cat_dir = os.path.join(train_dest, category)
        test_cat_dir = os.path.join(test_dest, category)
        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(test_cat_dir, exist_ok=True)
        
        for img in train_images:
            shutil.copy(os.path.join(cat_path, img), os.path.join(train_cat_dir, img))
            
        for img in test_images:
            shutil.copy(os.path.join(cat_path, img), os.path.join(test_cat_dir, img))
            
        print(f"Processed {category}: {len(train_images)} train, {len(test_images)} test")

print("Dataset splitting complete!")