import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Define dataset path
DATASET_PATH = "dataset/"
IMG_SIZE = (150, 150)
BATCH_SIZE = 5

# Verify dataset path exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path '{DATASET_PATH}' not found.")

# Apply data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    subset="validation"
)

# **Fix: Swap Labels if Needed**
expected_labels = {'cats': 1, 'dogs': 0}

if train_generator.class_indices['cats'] == 0:
    print("\nWARNING: Class labels are flipped! Fixing them in code...")
    train_generator.class_indices = expected_labels  # Force correct labels
    val_generator.class_indices = expected_labels
    train_generator.classes = np.abs(train_generator.classes - 1)  # Swap 0 ↔ 1
    val_generator.classes = np.abs(val_generator.classes - 1)  # Swap 0 ↔ 1

# Print dataset details
print("\nDataset Details")
print(f"Training set size: {train_generator.samples}")
print(f"Validation set size: {val_generator.samples}")

print(f"Class labels: {train_generator.class_indices}")  # {'cats': 1, 'dogs': 0}
print(f"Positive images (Cats): {sum(train_generator.classes == 1)}")
print(f"Negative images (Dogs): {sum(train_generator.classes == 0)}\n")

# **Optimized Model with Fewer Parameters**
base_model = keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base layers

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Reduces parameter count
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Print model structure
print("\nModel Summary")
model.summary()
print(f"\nTotal Trainable Parameters: {model.count_params()}")

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save trained model WITHOUT optimizer state to reduce file size
model.save("model.h5", include_optimizer=False)

# Print final accuracy
print("\nFinal Training Results")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")
print("Optimized model saved as model.h5")
