
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, BatchNormalizationV2, Flatten, Dropout, ZeroPadding3D
from globalConstants import TRAINING_EXAMPLE_DEPTH, TRAINING_EXAMPLE_HEIGHT, TRAINING_EXAMPLE_WIDTH

modelPath = "Models/model_10x240x135_5CCM_DR1_3"


print("Creating model...")
input_layer = Input(shape=(TRAINING_EXAMPLE_DEPTH,TRAINING_EXAMPLE_HEIGHT,TRAINING_EXAMPLE_WIDTH,1), name='input_layer')


ZeroPadding3D_1a = ZeroPadding3D(padding=(0,1,1))(input_layer)
Conv3D_1a = Conv3D(2, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_1a)
BaNo_1a = BatchNormalizationV2()(Conv3D_1a)
ZeroPadding3D_1b = ZeroPadding3D(padding=(0,1,1))(BaNo_1a)
Conv3D_1b = Conv3D(2, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_1b)
BaNo_1b = BatchNormalizationV2()(Conv3D_1b)
ZeroPadding3D_1c = ZeroPadding3D(padding=((0,0),(0,0),(0,1)))(BaNo_1b)
MaxPool3D_1 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(ZeroPadding3D_1c)

ZeroPadding3D_2a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_1)
Conv3D_2a = Conv3D(4, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_2a)
BaNo_2a = BatchNormalizationV2()(Conv3D_2a)
ZeroPadding3D_2b = ZeroPadding3D(padding=(0,1,1))(BaNo_2a)
Conv3D_2b = Conv3D(4, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_2b)
BaNo_2b = BatchNormalizationV2()(Conv3D_2b)
MaxPool3D_2 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(BaNo_2b)

ZeroPadding3D_3a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_2)
Conv3D_3a = Conv3D(6, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_3a)
BaNo_3a = BatchNormalizationV2()(Conv3D_3a)
ZeroPadding3D_3b = ZeroPadding3D(padding=(0,1,1))(BaNo_3a)
Conv3D_3b = Conv3D(6, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_3b)
BaNo_3b = BatchNormalizationV2()(Conv3D_3b)
MaxPool3D_3 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(BaNo_3b)

ZeroPadding3D_4a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_3)
Conv3D_4a = Conv3D(8, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_4a)
BaNo_4a = BatchNormalizationV2()(Conv3D_4a)
ZeroPadding3D_4b = ZeroPadding3D(padding=(0,1,1))(BaNo_4a)
Conv3D_4b = Conv3D(8, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_4b)
BaNo_4b = BatchNormalizationV2()(Conv3D_4b)
ZeroPadding3D_4c = ZeroPadding3D(padding=((0,0),(0,0),(0,1)))(BaNo_4b)
MaxPool3D_4 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(ZeroPadding3D_4c)

ZeroPadding3D_5a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_4)
Conv3D_5a = Conv3D(10, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_5a)
BaNo_5a = BatchNormalizationV2()(Conv3D_5a)
ZeroPadding3D_5b = ZeroPadding3D(padding=(0,1,1))(BaNo_5a)
Conv3D_5b = Conv3D(10, (1,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_5b)
BaNo_5b = BatchNormalizationV2()(Conv3D_5b)
MaxPool3D_5 = MaxPooling3D(pool_size=(1,3,3), strides=(1,3,3), padding="valid")(BaNo_5b)

Flatten_1 = Flatten()(MaxPool3D_5)
Dense_1 = Dense(units=56, activation="relu")(Flatten_1)
Dropout_1 = Dropout(0.5)(Dense_1)
output_layer = Dense(units=4, activation="softmax")(Dropout_1)




model = Model(inputs=input_layer, outputs=output_layer)

model.summary()

print("Saving model...")
model.save(modelPath)






'''
import tensorflow as tf
from keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Flatten, Layer, GlobalAveragePooling3D
from keras.models import Model
from globalConstants import TRAINING_EXAMPLE_DEPTH, TRAINING_EXAMPLE_HEIGHT, TRAINING_EXAMPLE_WIDTH
import numpy as np

# Beispielparameter
PATCH_SIZE = 15
NUM_PATCHES_HEIGHT = TRAINING_EXAMPLE_HEIGHT // PATCH_SIZE # 9 Patches in der Höhe
NUM_PATCHES_WIDTH = TRAINING_EXAMPLE_WIDTH // PATCH_SIZE # 16 Patches in der Breite
NUM_PATCHES = NUM_PATCHES_HEIGHT * NUM_PATCHES_WIDTH # 9 * 16 = 144 Patches pro Frame
EMBED_DIM = 16 # Dimension des Patch-Embeddings
NUM_FRAMES = TRAINING_EXAMPLE_DEPTH # Zeitdimension
NUM_HEADS = 2 # Anzahl der Attention-Heads
NUM_TRANSFORMER_LAYERS = 1 # Anzahl der Transformer-Layer
modelPath = "Models/model_10x240x135_transf_2heads_1tfl_avgpool_4"

# Eingabeschicht
input_layer = Input(shape=(NUM_FRAMES, TRAINING_EXAMPLE_HEIGHT, TRAINING_EXAMPLE_WIDTH, 1), name='input_layer')

# Patch-Extraktion und -Embedding

#def extract_patches_and_embed(frames):
#    # Reshape und flache die Patches
#    patches = tf.image.extract_patches(
#        images=frames,
#        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
#        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
#        rates=[1, 1, 1, 1],
#        padding='VALID'
#    )
#    # Patches flach machen und in Embeddings umwandeln
#    patches = tf.reshape(patches, [-1, NUM_FRAMES, NUM_PATCHES, PATCH_SIZE*PATCH_SIZE])
#    patch_embeddings = Dense(EMBED_DIM)(patches)
#    return patch_embeddings

def extract_patches_and_embed(frames):
    # Loop über die Frames und Patches für jedes Frame einzeln extrahieren
    patch_list = []
    for i in range(NUM_FRAMES):
        frame = frames[:, i] # Frame auswählen (Shape: (batch_size, height, width, channels))
        patches = tf.image.extract_patches(
            images=frame,
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Patches für dieses Frame in die Liste aufnehmen
        patch_list.append(patches)

    # Patches entlang der Zeitdimension zusammenführen (Shape: (batch_size, num_frames, num_patches, patch_size*patch_size))
    patches = tf.stack(patch_list, axis=1)
    # Patches flach machen und in Embeddings umwandeln
    patch_embeddings = Dense(EMBED_DIM)(patches)
    return patch_embeddings

# Positionsembeddings hinzufügen

#def add_position_embeddings(inputs):
#    pos_embeds = tf.Variable(tf.random.normal([1, NUM_FRAMES, NUM_PATCHES_HEIGHT, NUM_PATCHES_WIDTH, EMBED_DIM]))
#    return inputs + pos_embeds

class AddPositionEmbeddings(Layer):
    def __init__(self, num_frames, num_patches_height, num_patches_width, embed_dim, **kwargs):
        super(AddPositionEmbeddings, self).__init__(**kwargs)
        self.pos_embeds = self.add_weight(
            shape=(1, num_frames, num_patches_height, num_patches_width, embed_dim),
            initializer='random_normal',
            trainable=True,
            name='position_embeddings'
        )

    def call(self, inputs):
        return inputs + self.pos_embeds

# Transformer-Schicht
def transformer_block(inputs):
    x = LayerNormalization()(inputs)
    attention_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM)(x, x)
    x = x + attention_output
    x = LayerNormalization()(x)
    x = x + Dense(EMBED_DIM, activation='relu')(x)
    return x

# Extrahiere Patches und füge Positionsembeddings hinzu
patch_embeddings = extract_patches_and_embed(input_layer)
patch_embeddings = AddPositionEmbeddings(NUM_FRAMES, NUM_PATCHES_HEIGHT, NUM_PATCHES_WIDTH, EMBED_DIM)(patch_embeddings)

# Mehrere Transformer-Blöcke anwenden
x = patch_embeddings
for _ in range(NUM_TRANSFORMER_LAYERS):
    x = transformer_block(x)

# Flatten für Klassifizierung
#x = Flatten()(x)
x = GlobalAveragePooling3D()(x)
output_layer = Dense(units=4, activation='softmax')(x)

# Modell erstellen
model = Model(inputs=input_layer, outputs=output_layer)

# Zusammenfassung des Modells
model.summary()

print("Saving model...")
model.save(modelPath)

'''

