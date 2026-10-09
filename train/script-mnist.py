import tensorflow as tf
from tensorflow.keras import layers, models
import tf2onnx

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (28, 28) -> (28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# Build CNN model
# Named input
inputs = tf.keras.Input(
    shape=(28, 28, 1),
    name="input_layer"
)

x = layers.Conv2D(
    32, (3, 3),
    activation="relu"
)(inputs)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(
    64, (3, 3),
    activation="relu"
)(x)

x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.5)(x)

# Named output
outputs = layers.Dense(
    10,
    activation="softmax",
    name="output_layer"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="mnist_cnn"
)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

model.save("mnist.keras")