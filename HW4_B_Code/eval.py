import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Recompile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define dataset path
DATASET_PATH = "dataset/"
IMG_SIZE = (150, 150)
BATCH_SIZE = 5

# Load dataset (REMOVE validation split)
datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb"  # Ensure images are RGB
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print(f"Model Accuracy on Test Set: {accuracy:.2f}")
