import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
import requests
#from models.dinov2 import DinoV2Finetune
from difflib import SequenceMatcher
from torchvision import transforms
import sys
import tqdm
import tensorflow as tf
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)





def azure_ocr(image_path, cheese_names):
    if 'COMPUTER_VISION_KEY' in os.environ:
        subscription_key = os.environ['COMPUTER_VISION_KEY']
    else:
        print("\nSet the COMPUTER_VISION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
        sys.exit()

    if 'COMPUTER_VISION_ENDPOINT' in os.environ:
        endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
    else:
        print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
        sys.exit()

    analyze_url = endpoint + "computervision/imageanalysis:analyze"

    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    params = {
        "api-version": "2023-02-01-preview",
        "features": "Read",
        "language": "en",
        "gender-neutral-caption": "False"
    }
    response = requests.post(
        analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    description_lines = response.json()["readResult"]["content"].upper().split("\n")

    # Find the best match among cheese names for each line of description
    best_match_line = None
    best_match_cheese = None
    highest_similarity = 0

    for line in description_lines:
        # Find the best match among cheese names for the current line
        line_best_match = None
        line_highest_similarity = 0
        for cheese_name in cheese_names:
            similarity = SequenceMatcher(None, line, cheese_name).ratio()
            if similarity > line_highest_similarity:
                line_highest_similarity = similarity
                line_best_match = cheese_name

        # Check if the best match for this line is better than the overall best match
        if line_highest_similarity > highest_similarity:
            highest_similarity = line_highest_similarity
            best_match_line = line
            best_match_cheese = line_best_match
    return best_match_cheese, highest_similarity


@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg, test_path: str = "/Data/hala.gamouh/cheese_classification_challenge/dataset/test"):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    
    class_names = sorted(os.listdir(cfg.dataset.train_path))

    ocr_results = pd.DataFrame(columns=["id", "label","highest_similarity"])
    for i, batch in enumerate(test_loader):
        print("Batch: ", i, " / ", len(test_loader))
        images, image_names = batch
        images = images.to(device)
        for img_name in tqdm.tqdm(image_names,f"Processing images for batch {i}"):
            image_path = os.path.join(cfg.dataset.test_path, img_name + '.jpg')
            best_match, highest_similarity = azure_ocr(image_path, class_names)
            #print(f"Best match text for {img_name}: '{best_match}'")
            ocr_results = pd.concat([ocr_results, pd.DataFrame({"id": [img_name], "label": [best_match], "highest_similarity": [highest_similarity]})], ignore_index=True)


            #print(f"The prediction for {img_name} is {pred}")


    ocr_results.to_csv(f"{cfg.root_dir}/ocr_results.csv", index=False)


if __name__ == "__main__":
    create_submission()
