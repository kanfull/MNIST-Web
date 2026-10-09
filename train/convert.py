import tensorflow as tf

model = tf.keras.models.load_model("mnist.keras")

tf.saved_model.save(model, "mnist_saved_model")

#python -m tf2onnx.convert --saved-model mnist_saved_model --output model.onnx --opset 13