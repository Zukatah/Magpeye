import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, BatchNormalizationV2, Flatten, Dropout, ZeroPadding3D
from globalConstants import TRAINING_EXAMPLE_DEPTH, TRAINING_EXAMPLE_HEIGHT, TRAINING_EXAMPLE_WIDTH

modelPath = "Models/model_240x135_zp011_12c133111_bn_mp133133_zp011_12c133111_bn_zp011_12c233111_bn_mp143143_zp011_24c133111_bn_zp011_24c133111_bn_mp143143_zp011_36c133111_bn_mp155155_fl_den32_do20"


print("Creating model...")
input_layer = Input(shape=(TRAINING_EXAMPLE_DEPTH,TRAINING_EXAMPLE_HEIGHT,TRAINING_EXAMPLE_WIDTH,1), name='input_layer')

ZeroPadding3D_1 = ZeroPadding3D(padding=(0,1,1))(input_layer)
Conv3D_1 = Conv3D(12, (1,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_1)
BaNo_1 = BatchNormalizationV2()(Conv3D_1)
MaxPool3D_1 = MaxPooling3D(pool_size=(1,3,3), strides=(1,3,3), padding="valid")(BaNo_1)
ZeroPadding3D_2a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_1)
Conv3D_2a = Conv3D(12, (1,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_2a)
BaNo_2a = BatchNormalizationV2()(Conv3D_2a)
ZeroPadding3D_2b = ZeroPadding3D(padding=(0,1,1))(BaNo_2a)
Conv3D_2b = Conv3D(12, (2,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_2b)
BaNo_2b = BatchNormalizationV2()(Conv3D_2b)
MaxPool3D_2 = MaxPooling3D(pool_size=(1,4,3), strides=(1,4,3), padding="valid")(BaNo_2b)
ZeroPadding3D_3a = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_2)
Conv3D_3a = Conv3D(24, (1,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_3a)
BaNo_3a = BatchNormalizationV2()(Conv3D_3a)
ZeroPadding3D_3b = ZeroPadding3D(padding=(0,1,1))(BaNo_3a)
Conv3D_3b = Conv3D(24, (1,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_3b)
BaNo_3b = BatchNormalizationV2()(Conv3D_3b)
MaxPool3D_3 = MaxPooling3D(pool_size=(1,4,3), strides=(1,4,3), padding="valid")(BaNo_3b)
ZeroPadding3D_4 = ZeroPadding3D(padding=(0,1,1))(MaxPool3D_3)
Conv3D_4 = Conv3D(36, (1,3,3), strides=(1,1,1), activation="relu", padding="valid")(ZeroPadding3D_4)
BaNo_4 = BatchNormalizationV2()(Conv3D_4)
MaxPool3D_4 = MaxPooling3D(pool_size=(1,5,5), strides=(1,5,5), padding="valid")(BaNo_4)

Flatten_1 = Flatten()(MaxPool3D_4)
Dense_1 = Dense(units=16, activation="relu")(Flatten_1)
Dropout_1 = Dropout(0.2)(Dense_1)
output_layer = Dense(units=4, activation="softmax")(Dropout_1)





model = Model(inputs=input_layer, outputs=output_layer)


print("Saving model...")
model.save(modelPath)