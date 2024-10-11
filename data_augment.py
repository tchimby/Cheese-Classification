import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from albumentations import (Compose, RandomResizedCrop, Rotate, RandomBrightnessContrast, Normalize, Resize)
from albumentations.pytorch import ToTensorV2

data_dir = '/Data/hala.gamouh/ip_adapter_data'
output_dir = '/Data/hala.gamouh/augmented_data'
os.makedirs(output_dir, exist_ok=True)

# basic augmentations using torchvision
basic_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# advanced augmentations using albumentations
advanced_transforms = Compose([
    Resize(256, 256),
    RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    Rotate(limit=10),
    RandomBrightnessContrast(p=0.2),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
def numpy_to_tensor(image_np):
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(image_tensor)

#apply basic augmentations and save images
def augment_and_save_basic(image_path, output_dir, index, transform):
    image = Image.open(image_path).convert("RGB")
    augmented_image = transform(image)
    augmented_image_pil = transforms.ToPILImage()(augmented_image)
    augmented_image_pil.save(os.path.join(output_dir, f'basic_{index}.jpg'))

# apply advanced augmentations and save images
def augment_and_save_advanced(image_path, output_dir, index, transform):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    augmented = transform(image=image_np)
    augmented_image_np = augmented['image']
    augmented_image_tensor = numpy_to_tensor(augmented_image_np)
    augmented_image_pil = transforms.ToPILImage()(augmented_image_tensor)
    augmented_image_pil.save(os.path.join(output_dir, f'advanced_{index}.jpg'))

# process each cheese name folder
for cheese_name in os.listdir(data_dir):
    cheese_dir = os.path.join(data_dir, cheese_name)
    output_cheese_dir = os.path.join(output_dir, cheese_name)
    os.makedirs(output_cheese_dir, exist_ok=True)
    for idx, image_name in enumerate(tqdm(os.listdir(cheese_dir), desc=cheese_name)):
        image_path = os.path.join(cheese_dir, image_name)
        if not os.path.isfile(image_path):
            continue
        augment_and_save_basic(image_path, output_cheese_dir, f'basic_{idx}', basic_transforms)
        augment_and_save_advanced(image_path, output_cheese_dir, f'advanced_{idx}', advanced_transforms)
