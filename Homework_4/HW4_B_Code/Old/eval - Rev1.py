import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define dataset path
DATASET_PATH = "dataset/"
IMG_SIZE = (150, 150)
BATCH_SIZE = 5

# Load dataset
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

test_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print(f"Model Accuracy on Test Set: {accuracy:.2f}")
