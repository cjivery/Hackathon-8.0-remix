import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Load dataset (example: flowers)
(ds_train, ds_val), ds_info = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True,
)

# Preprocess
def format_example(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(format_example).batch(32).prefetch(1)
ds_val = ds_val.map(format_example).batch(32).prefetch(1)

# Build model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds_train, epochs=5, validation_data=ds_val)

# Save model
model.save('my_model.h5')
