import tensorflow as tf

# Define the custom L1Dist layer
class L1Dist(tf.keras.layers.Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Load model with the custom L1Dist layer
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"L1Dist": L1Dist}  # Register the custom layer
    )
    print("Model loaded successfully.")
    return model
