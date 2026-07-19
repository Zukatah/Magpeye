import argparse
from keras.models import load_model
import tensorflow as tf
import os


def load_and_convert_model(model_filename):
    models_dir = "Models"
    model_path = os.path.join(models_dir, model_filename)

    if not model_filename.endswith(".keras"):
        print("Error: The file name must end with .keras")
        return

    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.")
        return

    print("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model. Ensure it's a valid Keras model. Error: {e}")
    model.summary()

    print("Converting model...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        model_tflite = converter.convert()

        output_filename = model_filename.replace(".keras", ".tflite")
        output_path = os.path.join(models_dir, output_filename)
    except Exception as e:
        raise RuntimeError(f"Failed to convert the model to TFLite. Error: {e}")
        
    with open(output_path, "wb") as f:
        f.write(model_tflite)

    print(f"Model converted and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert a Keras model in the Models folder to TensorFlow Lite')
    parser.add_argument('--modelFilename', type=str, help='Filename of the Keras model (must end with .keras)')
    args = parser.parse_args()

    if args.modelFilename:
        load_and_convert_model(args.modelFilename)
    else:
        model_filename = input("Enter the filename of the Keras model (must end with .keras): ")
        load_and_convert_model(model_filename)


if __name__ == "__main__":
    main()