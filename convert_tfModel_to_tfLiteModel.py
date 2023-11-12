from keras.models import load_model
import tensorflow as tf

modelPath = "Models/model_dropout20_den32_den32_den32_stride133"

print("Loading model...")
model = load_model(modelPath)

print("Saving model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model = model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
model_tflite = converter.convert()
open(modelPath + "/model.tflite", "wb").write(model_tflite)