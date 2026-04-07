import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/waste_model.h5")

class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

def predict_image(image):
    img = image.resize((160,160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return class_names[np.argmax(score)], float(np.max(score))