import tensorflow as tf
from siamese_net.paper_code.siamese_net import *
import numpy as np


# ==========================================================
#  Loading and Create Siamese Model
# ==========================================================
def load_and_convert_model(path):
    # Load Siamese Model
    siamese_model = (SiameseNet(input_shape=(31, 1, 1))).load_saved_model(path)

    # Conversion Siamese Model
    converter = tf.lite.TFLiteConverter.from_keras_model(siamese_model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('siamese_model.tflite', 'wb') as f:
        f.write(tflite_model)


# ==========================================================
# Test TFL Siamese Model
# ==========================================================
def test_tflite_model():
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path='siamese_model.tflite')
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare sample input data (replace with actual data)
    input_shape_left = input_details[0]['shape']
    input_shape_right = input_details[1]['shape']
    input_data_left = np.random.rand(*input_shape_left).astype(np.float32)
    input_data_right = np.random.rand(*input_shape_right).astype(np.float32)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data_left)
    interpreter.set_tensor(input_details[1]['index'], input_data_right)

    # Invoke the model
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output:", output_data)


if __name__ == "__main__":
    model_path = '../siamese_net/results/paper_code/train_test/con_mqtt/siamese_model.h5'

    load_and_convert_model(model_path)
    test_tflite_model()