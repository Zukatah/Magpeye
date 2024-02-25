import numpy as np
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import consumingSetsOfFiles_listFiles
import csv
import tensorflow as tf
import argparse
# import sklearn.model_selection # TODO: check use of this method


train_ds = consumingSetsOfFiles_listFiles.lf_ds_train
val_ds = consumingSetsOfFiles_listFiles.lf_ds_val
val_ds_filename = consumingSetsOfFiles_listFiles.lf_ds_val_filename

modelName = ""
modelPath = ""
model = None
learningRate = 0.001
epochsCount = 10


def loadAndCompileModel():
    print("Loading model...")
    global model
    model = load_model(modelPath)
    print("Compiling model...")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learningRate), metrics=['acc']) # usually lr between 0.001 and 0.000005
    model.summary()
    

def trainModel():
    print("Fitting model...")
    model_checkpoint_callback = ModelCheckpoint('./Models/tmp/cp_{val_loss:.4f}_' + modelName, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')
    history = model.fit(train_ds, epochs=epochsCount, validation_data=val_ds, callbacks=[model_checkpoint_callback])
    print("Saving model...")
    model.save(modelPath)


def evaluateModel():
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
        y_pred.extend(preds)
        y_pred_classes.extend(np.argmax(preds, axis=1))
        file_pathes.extend(file_path_batch)
    print(classification_report(y_true, y_pred_classes))

    print("Saving information about files (training samples) that were classified correctly...")
    with open('NotesAndTools/CorrectlyLabeledFiles.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(file_pathes)):
            if y_true[i] == y_pred_classes[i]:
                writer.writerow([file_pathes[i].numpy(), y_true[i], y_pred_classes[i], int(y_pred[i][0]*100), int(y_pred[i][1]*100), int(y_pred[i][2]*100), int(y_pred[i][3]*100)])

    print("Saving information about files (training samples) that were classified incorrectly...")
    with open('NotesAndTools/MislabeledFiles.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(file_pathes)):
            if y_true[i] != y_pred_classes[i]:
                writer.writerow([file_pathes[i].numpy(), y_true[i], y_pred_classes[i], int(y_pred[i][0]*100), int(y_pred[i][1]*100), int(y_pred[i][2]*100), int(y_pred[i][3]*100)])


def main():
    global modelName
    global modelPath
    global learningRate
    global epochsCount

    parser = argparse.ArgumentParser(description='Load, train and evaluate models')
    parser.add_argument('--modelPath', type=str, help='Path to the Keras model directory')
    parser.add_argument('--learningRate', type=float, help='Learning Rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--evaluate', action='store_true', help='If flag is set, evaluate, else train.')
    args = parser.parse_args()

    if args.learningRate:
        learningRate = args.learningRate
    
    if args.epochs:
        epochsCount = args.epochs

    if args.modelPath:
        modelName = args.modelPath
    else:
        modelName = input("Enter the path to the Keras model directory: ")
    
    modelPath = "Models/" + modelName
    
    loadAndCompileModel()

    if args.evaluate:
        evaluateModel()
    else:
        trainModel()


if __name__ == "__main__":
    main()