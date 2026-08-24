import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1) PATHS
# =========================
source_dir = r'C:\Users\Saif AI\OneDrive - Irbid National University\Desktop\X-ray Images of Fractured and Healthy Bones\X-ray Imaging Dataset for Detecting Fractured vs. Non-Fractured Bones\Augmented Dataset'
split_dir = os.path.join(source_dir, "SplitDataset") # Result from splitting script

train_dir = os.path.join(split_dir, "train")
val_dir   = os.path.join(split_dir, "val")
test_dir  = os.path.join(split_dir, "test")

classes = ["Non-Fractured", "Fractured"]

# =========================
# 2) CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 4       # Initial training for head only
EPOCHS_FINE = 4       # Fine-tuning phase
SEED = 42

# =========================
# 3) DATA GENERATORS
# =========================
# Using MobileNetV2 preprocess_input is better than manual 1/255 rescaling
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    classes=classes,
    shuffle=True,
    seed=SEED
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    classes=classes,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    classes=classes,
    shuffle=False
)

# =========================
# 4) CLASS WEIGHTS (Calculated from training data only)
# =========================
y_train = train_gen.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# =========================
# 5) MODEL (Transfer Learning)
# =========================
def build_model(input_shape=(224, 224, 3)):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False  # Phase 1: Freeze base model

    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model, base

model, base_model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# =========================
# 6) CALLBACKS
# =========================
callbacks_head = [
    ModelCheckpoint("best_head.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
]

# =========================
# 7) TRAIN - HEAD
# =========================
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head,
    class_weight=class_weight_dict,
    verbose=1
)

# =========================
# 8) FINE-TUNING
# =========================
# Unfreeze the base model layers
base_model.trainable = True

# Choose number of layers to unfreeze (last 40 layers)
fine_tune_at = len(base_model.layers) - 40
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Very small Learning Rate for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

callbacks_fine = [
    ModelCheckpoint("best_finetuned.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
]

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine,
    class_weight=class_weight_dict,
    verbose=1
)

# =========================
# 9) EVALUATION (Separate Test Set)
# =========================
print("\nEvaluating best finetuned model on TEST...")
best_model = tf.keras.models.load_model("best_finetuned.keras")
test_gen.reset()
results = best_model.evaluate(test_gen, verbose=1)
print("Test results:", dict(zip(best_model.metrics_names, results)))

# Predictions
test_gen.reset()
probs = best_model.predict(test_gen, verbose=1).ravel()
y_pred = (probs > 0.5).astype(int)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# =========================
# 10) PLOT HISTORY (Combine Head + Fine-tuning data)
# =========================
def merge_history(h1, h2):
    out = {}
    for k in h1.history.keys():
        out[k] = h1.history[k] + h2.history.get(k, [])
    return out

hist = merge_history(history_head, history_fine)

plt.figure(figsize=(15, 5))
for i, metric in enumerate(["loss", "accuracy", "auc"]):
    plt.subplot(1, 3, i+1)
    plt.plot(hist[metric], label="Train")
    plt.plot(hist[f"val_{metric}"], label="Val")
    plt.title(metric.capitalize())
    plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 11) SAVE FINAL MODEL (Current Directory)
# =========================
# Get the directory where this script is currently running
current_dir = os.getcwd()

# Define the full path including the model's filename
final_model_path = os.path.join(current_dir, "best_fracture_model.keras")

# Save the loaded best model to the same folder
best_model.save(final_model_path)

print(f"\nSUCCESS ✅ Final model saved locally to: {final_model_path}")
