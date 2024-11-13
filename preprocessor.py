from PIL import Image
import numpy as np

def preprocess_input(image):
    # Convert image to RGB
    img = Image.open(image).convert('RGB')
    # Resize to model's expected input size
    img = img.resize((224, 224))  # Adjust dimensions as per your model
    # Normalize pixel values
    img_array = np.array(img) / 255.0
    return img_array
