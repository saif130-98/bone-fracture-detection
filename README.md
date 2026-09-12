🦴 DeepBone: X-Ray Fracture Detection AI
DeepBone is an end-to-end computer vision pipeline and interactive web application designed to analyze medical X-ray images and classify them as Fractured or Non-Fractured. Leveraging a two-phase transfer learning approach with MobileNetV2, this tool acts as an AI diagnostic assistant with exceptional real-world accuracy.

✨ Project Highlights
High-Accuracy Classification: Achieves 99% overall accuracy on unseen test data.

Clinical Safety First: Out of 1,396 actual fractures in the test set, the model only missed 6 (False Negatives), making it highly reliable as a preliminary screening tool.

Two-Phase Transfer Learning: Utilizes a frozen MobileNetV2 base for initial feature extraction, followed by targeted fine-tuning of the top 40 layers to adapt to specific X-ray textures.

Robust Data Pipeline: Implements dynamic data augmentation (rotation, shifting, flipping, brightness adjustments) and automated class weighting to ensure balanced, generalized learning.

Interactive UI: Includes a custom Streamlit web application (app.py) for drag-and-drop inference, featuring confidence progress bars and medical disclaimers.

📊 Model Performance
The model was evaluated on a perfectly balanced test set of 2,792 images (1,396 Fractured, 1,396 Non-Fractured).

Accuracy: 99%

Precision (Fractured): 0.99

Recall (Fractured): 1.00

F1-Score: 0.99

Confusion Matrix Breakdown:

True Positives (Correctly identified fractures): 1,390

True Negatives (Correctly identified healthy bones): 1,380

False Positives (False alarms): 16

False Negatives (Missed fractures): 6

🛠️ Technical Architecture
Base Model: Pre-trained MobileNetV2 (ImageNet weights)

Custom Head: GlobalAveragePooling2D ➡️ Dropout(0.5) ➡️ Dense(128, ReLU) ➡️ Dense(1, Sigmoid)

Input Shape: (224, 224, 3)

Optimization Strategy:

Phase 1 (Head Training): Adam Optimizer (lr=1e-4), 3 Epochs.

Phase 2 (Fine-Tuning): Adam Optimizer (lr=1e-5), 3 Epochs.

Callbacks: ModelCheckpoint (saves best validation AUC), EarlyStopping, ReduceLROnPlateau.
