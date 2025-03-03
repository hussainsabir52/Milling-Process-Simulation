import tensorflow as tf

model = tf.keras.models.load_model("model1_cnn_base.keras")


model.summary()


for layer in model.layers:
    print("Layer:", layer.name)
    weights = layer.get_weights()
    for weight in weights:
        print(weight.shape)
