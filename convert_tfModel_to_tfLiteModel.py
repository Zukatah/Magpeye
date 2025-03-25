import argparse
from keras.models import load_model
import tensorflow as tf
import os


def load_and_convert_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model. Ensure it's a valid Keras model. Error: {e}")
    model.summary()

    print("Saving model...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        model_tflite = converter.convert()
    except Exception as e:
        raise RuntimeError(f"Failed to convert the model to TFLite. Error: {e}")
        
    
    # output_path = model_path + "/model.tflite"
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = os.path.join(os.path.dirname(model_path), f"{base_name}.tflite")
    # open(output_path, "wb").write(model_tflite)
    with open(output_path, "wb") as f:
        f.write(model_tflite)
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