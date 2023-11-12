import numpy as np
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import consumingSetsOfFiles_listFiles
import csv
import tensorflow as tf
# import sklearn.model_selection # TODO: check use of this method

train_ds = consumingSetsOfFiles_listFiles.lf_ds_train
val_ds = consumingSetsOfFiles_listFiles.lf_ds_val
val_ds_filename = consumingSetsOfFiles_listFiles.lf_ds_val_filename

modelPath = "Models/model_c3d233133_mp3d122122_c3d233111_mp3d122122_c3d233111_mp3d122122_c3d233111_mp3d122122_den24_do40"


print("Loading model...")
model = load_model(modelPath)

print("Compiling model...")
model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['acc']) # usually lr between 0.001 and 0.000005
model.summary()
model_checkpoint_callback = ModelCheckpoint('./tmp/checkpoint', save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

print("Fitting model...")
history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[model_checkpoint_callback])

print("Saving model...")
model.save(modelPath)

'''
print("Evaluating model...")
y_true = []
y_pred = []
y_pred_classes = []
file_pathes = []
for image_batch, label_batch, file_path_batch in val_ds_filename:
    trues = label_batch.numpy()
    trues = np.argmax(trues, axis=1)
    y_true.extend(trues.tolist())
    preds = model.predict(image_batch)
    y_pred.append(preds)
    y_pred_classes.extend(np.argmax(preds, axis=1))
    file_pathes.extend(file_path_batch)
print(classification_report(y_true, y_pred_classes))
'''

# print("Output the classes of some training samples and the prediction of the model:")
# print("y_true", y_true[:100])
# print("y_pred", y_pred_classes[:100]) #print("y_pred", y_pred[:100])

'''
print("Saving information about files (training samples) that were classified incorrectly...")
with open('MislabeledFiles.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(file_pathes)):
        if y_true[i] != y_pred_classes[i]:
            writer.writerow([file_pathes[i].numpy(), y_true[i], y_pred_classes[i]])
'''