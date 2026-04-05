import os
from PIL import Image
import numpy as np

data_dir = "D:/dataset/PetImages"
categories = ["Cat", "Dog"]
img_size = 128

X = []
y = []

for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((img_size, img_size))
            img_array = np.array(img)
            
            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Loi file {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)