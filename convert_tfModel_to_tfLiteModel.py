import argparse
from keras.models import load_model
import tensorflow as tf


def load_and_convert_model(model_path):
    print("Loading model...")
    model = load_model(model_path)
    model.summary()

    print("Saving model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    model_tflite = converter.convert()
    
    output_path = model_path + "/model.tflite"
    open(output_path, "wb").write(model_tflite)
    print(f"Model converted and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Load and convert Keras model to TensorFlow Lite')
    parser.add_argument('--modelPath', type=str, help='Path to the Keras model directory')

    args = parser.parse_args()

    if args.modelPath:
        load_and_convert_model(args.modelPath)
    else:
        model_path = input("Enter the path to the Keras model directory: ")
        load_and_convert_model(model_path)


if __name__ == "__main__":
    main()