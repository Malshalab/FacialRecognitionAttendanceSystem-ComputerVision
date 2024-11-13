from utils.model_loader import load_model

# Define model path
model_path = "../model/siamese_model.h5"

# Load the model
model = load_model(model_path)

# Print confirmation
print("Model loaded and ready for predictions!")
