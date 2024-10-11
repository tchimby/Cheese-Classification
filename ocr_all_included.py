import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from difflib import SequenceMatcher
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import sys
from difflib import SequenceMatcher
import requests

cheese_names = [
    "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE",
    "MORBIER", "SAINT-NECTAIRE", "POULIGNY SAINT-PIERRE", "ROQUEFORT", "COMTÉ",
    "CHÈVRE", "PECORINO", "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE",
    "PARMESAN", "SAINT-FÉLICIEN", "MONT D’OR", "STILTON", "SCAMORZA", "CABECOU",
    "BEAUFORT", "MUNSTER", "CHABICHOU", "TOMME DE VACHE", "REBLOCHON",
    "EMMENTAL", "FETA", "OSSAU-IRATY", "MIMOLETTE", "MAROILLES", "GRUYÈRE",
    "MOTHAIS", "VACHERIN", "MOZZARELLA", "TÊTE DE MOINES", "FROMAGE FRAIS"
]

def azure_ocr(image_path, cheese_names) :
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
    return best_match_cheese


def preprocess_image_cv(image_path):
    """Apply a series of preprocessing steps to the image using OpenCV."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img.shape[0] > 2000 or img.shape[1] > 2000:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Denoise image
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

    # Increase contrast
    img = cv2.equalizeHist(img)

    # Apply adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return img

def ocr_image_tesseract(image_path):
    """Extract text from an image using Tesseract OCR with pre-processing."""
    preprocessed_image = preprocess_image_cv(image_path)
    preprocessed_image_pil = Image.fromarray(preprocessed_image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image_pil, config=custom_config)
    return text.strip().upper()

def ocr_image_easyocr(image_path):
    """Extract text from an image using EasyOCR."""
    reader = easyocr.Reader(['en']) 
    results = reader.readtext(image_path, detail=0)
    return ' '.join(results).upper()

def get_best_match(text, names, threshold=0.2):
    """Find the best match for the given text from a list of names."""
    best_match = None
    highest_similarity = 0
    for name in names:
        similarity = SequenceMatcher(None, text, name).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name
    return best_match, similarity #if highest_similarity >= threshold else None

def ocr_image(image_path):
    """Extract text from an image using Tesseract OCR with pre-processing."""
    preprocessed_image = preprocess_image_cv(image_path)
    preprocessed_image_pil = Image.fromarray(preprocessed_image)
    custom_config = r'--oem 3 --psm 6'  # Tesseract configuration
    text = pytesseract.image_to_string(preprocessed_image_pil, config=custom_config)
    return text.strip().upper()

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.exceptions import HttpResponseError


key = "d09dc6bea8e945169afebcc6cfc073ec"
endpoint = "https://computervisionmodal473v.cognitiveservices.azure.com/"
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

def ocr_image_azure(image_path):
    with open(image_path, "rb") as image_data:
        try:
            result = client.begin_recognize_printed_text(image_data=image_data)
            return "\n".join([line.text for line in result.result()])
        except HttpResponseError as err:
            print("Azure Text Analytics error:", err)
            return ""

def main(image_path):
    ocr_result = ocr_image_azure(image_path)
    print("OCR result from Azure Text Analytics:", ocr_result)

if __name__ == "__main__":
    image_path = "/Data/hala.gamouh/cheese_classification_challenge/dataset/test/0LjeckXemjZpd90.jpg"
    client = authenticate_client()
    main(image_path)