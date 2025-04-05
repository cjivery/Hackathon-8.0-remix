from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

# Authenticate using the Kaggle API key
api = KaggleApi()
api.authenticate()

# Download the dataset
api.dataset_download_files('emmarex/plantdisease', path='.', unzip=True)

# Set dataset directory
dataset_dir = 'C:\\Users\\sasuk\\PycharmProjects\\Hackathon-8.0-remix\\archive\\PlantVillage'  # Replace with actual path

# Load the dataset from directory
full_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(256, 256),
    batch_size=32,
    label_mode='int',
    validation_split=0.2,  # Automatically splits the dataset
    subset="training",  # This will use the training data from the split
    seed=123  # To ensure reproducibility
)

val_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(256, 256),
    batch_size=32,
    label_mode='int',
    validation_split=0.2,  # Use the same split value for validation
    subset="validation",  # This will use the validation data from the split
    seed=123  # Same seed for consistency
)

# Cache and prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = full_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Build the model
model = models.Sequential([
    layers.InputLayer(input_shape=(256, 256, 3)),
    layers.Rescaling(1./255),  # Normalize image pixels to [0, 1]
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(full_dataset.class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Save the trained model
model.save('plant_disease_model.h5')
