import random
import os
import json
import requests
import sys
from openai import AzureOpenAI

api_key = os.getenv("AZURE_OPENAI_API_KEY")
base_directory = "/Data/hala.gamouh/dalle3_data"
generator = AzureOpenAI(api_version="2024-02-01", azure_endpoint="https://openai-modal473v.openai.azure.com/", api_key=api_key)
cheese_descriptions = {
    "BRIE DE MELUN": "Soft, creamy cheese with a white, bloomy rind and a pale yellow center.",
    "CAMEMBERT": "Soft, creamy cheese, similar in appearance to Brie but typically smaller, with a white, velvety rind.",
    "EPOISSES": "Soft-paste cheese with a distinctive bright orange rind, known for its glossy, sticky surface.",
    "FOURME D’AMBERT": "Semi-hard, blue-veined cheese with a creamy texture.",
    "RACLETTE": "Semi-hard, pale yellow cheese known for its smooth, meltable texture.",
    "MORBIER": "Semi-soft cheese characterized by a distinctive black ash layer in the middle.",
    "SAINT-NECTAIRE": "Soft and supple with a smooth, pale yellow rind.",
    "POULIGNY SAINT-PIERRE": "Tall, pyramid-shaped goat cheese with a crumbly texture and natural rind.",
    "ROQUEFORT": "Crumbly blue cheese with prominent blue veins.",
    "COMTÉ": "Firm cheese with a smooth, golden yellow rind and visible marbling.",
    "CHÈVRE": "Ranges from bright white, fresh and soft to aged and firm with a rough rind.",
    "PECORINO": "Hard, aged cheese with a rough, golden-brown rind.",
    "NEUFCHATEL": "Heart-shaped, soft cheese with a soft, white, bloomy rind.",
    "CHEDDAR": "Ranges from white to deep orange, firm, with a crumbly texture.",
    "BÛCHETTE DE CHÈVRE": "Small, log-shaped, firm goat cheese with a textured, mold-ripened rind.",
    "PARMESAN": "Hard, gritty, aged with a deep golden crust.",
    "SAINT-FÉLICIEN": "Soft, creamy cheese often encased in a wooden box, with a smooth, light rind.",
    "MONT D’OR": "Creamy, encircled with spruce bark, showing a gooey texture when melted.",
    "STILTON": "Blue cheese with a creamy base and extensive blue veining.",
    "SCAMORZA": "Pear-shaped with a smooth, semi-firm texture, often smoked to a golden hue.",
    "CABECOU": "Small, round, soft goat cheese with a mild rind.",
    "BEAUFORT": "Firm, with a smooth, shiny, pale yellow rind.",
    "MUNSTER": "Soft, with a shiny, bright orange rind.",
    "CHABICHOU": "Small, cylindrical, firm with a fine, wrinkled rind.",
    "TOMME DE VACHE": "Thick rind, firm texture, varies in color from pale yellow to gray.",
    "REBLOCHON": "Soft, with a washed, orange rind and creamy interior.",
    "EMMENTAL": "Firm, pale yellow with large holes throughout.",
    "FETA": "White, crumbly texture, traditionally cubed or crumbled.",
    "OSSAU-IRATY": "Smooth, slightly oily texture with a natural, hard rind.",
    "MIMOLETTE": "Bright orange, hard cheese with a smooth, round shape.",
    "MAROILLES": "Soft, with a moist, bright orange rind.",
    "GRUYÈRE": "Rich, creamy, pale yellow with small holes and a smooth rind.",
    "MOTHAIS": "Creamy, encased in a natural rind with a slightly wrinkled texture.",
    "VACHERIN": "Rich, creamy, encased in spruce bark, with a melty texture.",
    "MOZZARELLA": "Soft, moist, white, and usually in ball form.",
    "TÊTE DE MOINES": "Semi-hard, shaved into thin rosettes, revealing a pale yellow core.",
    "FROMAGE FRAIS": "Soft, creamy, white, similar in appearance to yogurt."
}


# randomly drawn prompts :
variations = {
    "sliced": 11, "cut into cubes": 3, "in its whole form": 11, "packaged": 2,
    "grated": 2, "melted": 2, "accompanied by wine": 2,
    "featured in a sandwich": 2, "as part of a salad": 2, "" : 5, "with a knife": 5, "with charcuterie" : 2, "with a fork": 5, "with a spoon": 5, "in a well known recipe of it": 5, "" : 10
}

human = {
    "A person holding it": 5, "A person eating it": 10, "A person preparing it": 8, "A person serving it": 7,
    "A person cutting it": 5, "A person wrapping it": 1, "no human": 45
}

angle = {
    "from above": 10, "close-up": 10, "from the side": 5, "in a panoramic view": 2,
    "in a zoomed-in shot": 3
}

etiquette = {
    "with a label of the cheese name": 10, "with a label of the cheese origin": 5,
    "with a label of the cheese type": 5, "" : 15
}

place = {
    "in a kitchen": 10, "in a dining room": 5, "in a market": 5, "in a grocery store": 2,
    "in a cheese factory": 5, "in a storage room" :5, "": 15
}

def weighted_choice(choices):
    total = sum(choices.values())
    r = random.uniform(0, total)
    upto = 0
    for key, weight in choices.items():
        if upto + weight >= r:
            return key
        upto += weight


def generate_prompts(num_prompts,cheese):
    generated_prompts = []
    desc = cheese_descriptions[cheese]
    for _ in range(num_prompts):
        var = weighted_choice(variations)
        hum = weighted_choice(human)
        ang = weighted_choice(angle)
        eti = weighted_choice(etiquette)
        pla = weighted_choice(place)
    
        if var == "in a well known recipe of it" :
            prompt = f"An image of {cheese} {var}"
        elif cheese =='Raclette':
            if var == "melted" or var == "in its whole form":
                prompts = f"An image of {cheese} {var}"
            else :
                prompt = f"An image of {cheese}"
        else : prompt = f"An image of {cheese} , which is a {desc}, {var}, {eti}, seen {ang} with a {hum} {pla}"
        generated_prompts.append(prompt)
    return generated_prompts

def download_and_save_image(url, save_path):
    response = requests.get(url)
    response.raise_for_status() 
    with open(save_path, 'wb') as f:
        f.write(response.content)
        
for cheese, description in cheese_descriptions.items():
    cheese_directory = os.path.join(base_directory, cheese)
    os.makedirs(cheese_directory, exist_ok=True)  
    cheese_prompts = generate_prompts(10,cheese)
    
    for i, prompt in enumerate(cheese_prompts):
        print(f"Generating image for prompt: {prompt}")
        try :
            result = generator.images.generate(model="dall-e-3", prompt=prompt, n=1)
            image_url = json.loads(result.model_dump_json())['data'][0]['url']
            file_path = os.path.join(cheese_directory, f"{cheese.replace(' ', '_')}_{i}.png")
            download_and_save_image(image_url, file_path)
        except Exception as e:
            prompt = f"An image of {cheese} , which is a {description}"
            result = generator.images.generate(model="dall-e-3", prompt=prompt, n=1)
            image_url = json.loads(result.model_dump_json())['data'][0]['url']
            file_path = os.path.join(cheese_directory, f"{cheese.replace(' ', '_')}_{i}.png")
            download_and_save_image(image_url, file_path)

print("Images have been successfully generated and saved.")

