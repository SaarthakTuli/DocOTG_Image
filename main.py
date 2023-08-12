from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.model = tf.keras.models.load_model(
                "Final_model_vgg19.h5")
        return cls._instance

    def get_model(self):
        return self.model


app = Flask(__name__)
model_loader = ModelLoader()


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/image', methods=["POST"])
def image():
    model = model_loader.get_model()

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

    data = {
        "result": predicted_class
    }

    return f"{predicted_class}"

# main driver function
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
