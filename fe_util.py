import tensorflow as tf

model = tf.keras.models.load_model("gs://deit-tf/deit_tiny_patch16_224")
model.pre_logits = True

inputs = tf.keras.Input((224, 224, 3))
outputs = model(inputs)
model_fe = tf.keras.Model(inputs, outputs)

print(model.pre_logits)
print(model_fe(tf.ones((2, 224, 224, 3)))[0].shape)
