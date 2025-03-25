import tensorflow as tf
from tensorflow import keras

# Altes Modell laden
model = tf.keras.models.load_model('Models/model_10x240x135_5CCM_DR1_3')

# Im neuen `.keras`-Format speichern
model.save('Models/model_10x240x135_5CCM_DR1_3.h5', save_format='h5')