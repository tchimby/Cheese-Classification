# Overview:  
The project involves creating a classifier for cheese types when limited annotated data is available. To solve this, we generated synthetic training data using text-to-image models and employed advanced OCR techniques to enhance the generated images. The final model classified 37 cheese categories with a precision of 63\%.

# Approach:  

- Synthetic Data Generation:  
  Due to limited real-world annotated cheese datasets, synthetic images were generated using the following models:
  - Text-to-Image Models: DALL-E 3, Imagen, MidJourney, Stable Diffusion (SD 1.5, SD XL, Turbo, Lighting), IF, Pixart, etc.
  - OCR and DreamBooth: Used to enhance image quality and ensure relevant features for cheese categories were captured.
  - Prompt Engineering: Carefully crafted prompts ensured better diversity and precision in the generated synthetic data.

- Main Model:  
  The classifier used dinov2 (vision transformer) as the main model, fine-tuned to adapt to the variability of cheese images. The model addressed the gap between the synthetic training set and the real-world validation set.

# Validation and Testing:  
The provided validation and test sets helped evaluate model performance. Despite challenges from the distribution gap, the model achieved 63### precision across 37 cheese categories.

# Results:  

- Precision: 63 \% across 37 cheese categories.  
- Synthetic data generation improved, but the gap between training and validation data required additional fine-tuning.

# Future Work:  

- Improve synthetic image generation to reduce distribution differences.  
- Experiment with new prompt engineering strategies for greater data diversity.  
- Test alternative or ensemble models to enhance accuracy.
