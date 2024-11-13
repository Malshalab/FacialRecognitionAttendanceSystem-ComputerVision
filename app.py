from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from utils.model_loader import load_model

# Initialize Flask app
app = Flask(__name__)

# Model loading
MODEL_PATH = '../model/final_embedding_model.h5'

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    print(f"Request files: {request.files}")

    # Check if the request contains 'anchor' and 'validation' files
    if 'anchor' not in request.files or 'validation' not in request.files:
        return jsonify({"error": "Both anchor and validation images are required"}), 400

    anchor = request.files['anchor']
    validation = request.files['validation']

    if anchor.filename == '' or validation.filename == '':
        return jsonify({"error": "Both files must be selected"}), 400

    try:
        print(f"Processing files: {anchor.filename}, {validation.filename}")

        # Process anchor image
        anchor_img = Image.open(anchor).convert('RGB')
        anchor_img = anchor_img.resize((100, 100))
        anchor_array = np.array(anchor_img) / 255.0
        anchor_array = np.expand_dims(anchor_array, axis=0)

        # Process validation image
        validation_img = Image.open(validation).convert('RGB')
        validation_img = validation_img.resize((100, 100))
        validation_array = np.array(validation_img) / 255.0
        validation_array = np.expand_dims(validation_array, axis=0)

        # Generate embeddings
        anchor_embedding = model.predict(anchor_array)
        validation_embedding = model.predict(validation_array)

        # Compute distance
        distance = np.linalg.norm(anchor_embedding - validation_embedding)
        is_match = distance < 14  # Example threshold

        # Convert NumPy types to Python native types
        return jsonify({
            "distance": float(distance),
            "is_match": bool(is_match)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500




# Debugging test endpoint
@app.route('/test-upload', methods=['POST'])
def test_upload():
    # Print all files received in the request
    print(f"Request files: {request.files}")
    return jsonify({"received_files": list(request.files.keys())})

# Run the Flask app
if __name__ == '__main__':
    # Bind to 0.0.0.0 to make it accessible from other devices on the same network
    app.run(host='0.0.0.0', port=5000)
