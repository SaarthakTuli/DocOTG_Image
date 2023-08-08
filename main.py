from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)


# @app.route('/')
# def hello_world():
#     return 'Hello World'


@app.route('/image', methods=["POST"])
def image():
    # Loading Model
    model = tf.keras.models.load_model(
        "Final_model_vgg19.h5")

    # Retrieve the Base64-encoded image from the JSON payload
    data = request.get_json()
    encoded_image = data.get("image")

    # Decode the Base64 string to get the image data
    image_data = base64.b64decode(encoded_image)

    # Convert image data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    # Resize to 224x244 and converting to RGB mode
    image = image.convert('RGB')
    image = image.resize((224, 224))

    # Normalize pixel values to the range [0, 1]
    image_array = np.array(image) / 255.0

    # Expand the dimensions to create a batch of size 1
    input_data = np.expand_dims(image_array, axis=0)

    # Make the prediction
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions[0])
    return f"Predicted Class: {predicted_class}"


# @app.route('/knn')
# def knn():
#     return "<h1>KNN</h1>"


# @app.route('/svm')
# def svm():
#     return "<h1>SVM</h1>"


# main driver function
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
