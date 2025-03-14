import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report
import os

# Ensure model file exists
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Error: model.h5 not found! Train the model first.")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# **Fix: Compile the model before evaluation**
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define dataset path (using existing training dataset)
DATASET_PATH = "dataset/"
IMG_SIZE = (150, 150)
BATCH_SIZE = 5

# Load dataset for evaluation
datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False  # Keep labels aligned with predictions
)

# Ensure dataset is not empty
if test_generator.samples == 0:
    raise ValueError("No images found. Please check the dataset path.")

# Count cat and dog images
cat_count = sum(test_generator.classes == 1)  # Count of cat images
dog_count = sum(test_generator.classes == 0)  # Count of dog images

# Print dataset details
print("\nDataset Details")
print(f"Total images: {test_generator.samples}")
print(f"Class labels BEFORE correction: {test_generator.class_indices}")

# **Fix: Force class labels to match `train.py`**
expected_labels = {'cats': 1, 'dogs': 0}

if test_generator.class_indices != expected_labels:
    print("\nWARNING: Class labels are flipped! Fixing them in code...")
    test_generator.class_indices = expected_labels  # Force correct labels
    test_generator.classes = np.abs(test_generator.classes - 1)  # Swap 0 ↔ 1

print(f"Class labels AFTER correction: {test_generator.class_indices}")
print(f"Found {cat_count} cat images.")
print(f"Found {dog_count} dog images.")

# **Evaluate model**
loss, accuracy = model.evaluate(test_generator)
print(f"\nModel Evaluation Results")
print(f"Model Accuracy: {accuracy:.2f}")

# Get true labels and predictions
y_true = test_generator.classes  # Actual labels (0 = dog, 1 = cat)
y_pred_prob = model.predict(test_generator)
y_pred = np.round(y_pred_prob).astype(int)  # Convert probabilities to binary labels

# Print classification report (Precision, Recall, F1-score)
print("\nClassification Report")
print(classification_report(y_true, y_pred))

# Print individual predictions
print("\nPredictions (True_Class vs Predicted_Label)")
for i, filename in enumerate(test_generator.filenames):
    print(f"{filename}: True_Class = {y_true[i]}, Predicted = {y_pred[i][0]}")
