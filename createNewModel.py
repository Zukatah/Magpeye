import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, BatchNormalizationV2, Flatten, Dropout, ZeroPadding3D

modelPath = "Models/model_dropout50_den32_den32"


print("Creating model...")
input_layer = Input(shape=(5,480,270,1), name='input_layer')

ZeroPadding3D_1a = ZeroPadding3D(padding=(0,1,1))(input_layer)
Conv3D_1a = Conv3D(32, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_1a)
MaxPool3D_1 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(Conv3D_1a)

ZeroPadding3D_2a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_1)
Conv3D_2a = Conv3D(32, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_2a)
MaxPool3D_2 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(Conv3D_2a)

ZeroPadding3D_3a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_2)
Conv3D_3a = Conv3D(32, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_3a)
MaxPool3D_3 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(Conv3D_3a)

ZeroPadding3D_4a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_3)
Conv3D_4a = Conv3D(64, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_4a)
MaxPool3D_4 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid")(Conv3D_4a)

BatchNormalizationV2_1 = BatchNormalizationV2()(MaxPool3D_4)
Flatten_1 = Flatten()(BatchNormalizationV2_1)
Dense_1 = Dense(units=32, activation="relu")(Flatten_1)
Dropout_1 = Dropout(0.2)(Dense_1)
Dense_2 = Dense(units=32, activation="relu")(Dropout_1)
Dropout_2 = Dropout(0.2)(Dense_2)
output_layer = Dense(units=4, activation="softmax")(Dropout_2)

model = Model(inputs=input_layer, outputs=output_layer)


print("Saving model...")
model.save(modelPath)