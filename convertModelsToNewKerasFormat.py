import tensorflow as tf
from tensorflow import keras

# Altes Modell laden
model = tf.keras.models.load_model('Models/model_10x240x135_4CCM_CM_MorePooling_Nr1.h5')

# Im neuen `.keras`-Format speichern
model.save('Models/model_10x240x135_4CCM_CM_MorePooling_Nr1.keras', save_format='keras')