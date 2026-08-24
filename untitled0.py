import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# --- 1. Define Paths ---
# The path where you saved your final model
model_path = r"C:\Users\Saif AI\OneDrive - Irbid National University\Desktop\X-ray project\best_fracture_model.keras"

# ⚠️ CHANGE THIS to the exact path of the specific X-ray image you want to test
image_path = r"C:\Users\Saif AI\OneDrive - Irbid National University\Desktop\image\OIP.webp"

# --- 2. Load the Model ---
print("Loading model...")
model = tf.keras.models.load_model(model_path)

# --- 3. Process the Image ---
print("Processing image...")
# Load the image and resize it to match your training IMG_SIZE (270x270)
img = image.load_img(image_path, target_size=(224, 224))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add a batch dimension (models expect batches, so we make it a batch of 1)
# Shape changes from (270, 270, 3) to (1, 270, 270, 3)
img_batch = np.expand_dims(img_array, axis=0)

# Apply the exact same MobileNetV2 preprocessing used during training
img_preprocessed = preprocess_input(img_batch)

# --- 4. Make the Prediction ---
print("Making prediction...")
prediction = model.predict(img_preprocessed)
probability = prediction[0][0] # Get the raw probability from the sigmoid layer

# --- 5. Print the Results ---
print("\n" + "="*30)
if probability > 0.5:
    print("🦴 Prediction: FRACTURED")
    print(f"Confidence: {(probability * 100):.2f}%")
else:
    print("✅ Prediction: NON-FRACTURED (Healthy)")
    print(f"Confidence: {((1 - probability) * 100):.2f}%")
print("="*30 + "\n")