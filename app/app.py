import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("waste_model.h5")

class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

def predict_image(image):
    img = image.resize((160,160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()

    label = class_names[np.argmax(score)]
    confidence = float(np.max(score))

    # FIXED logic
    if label in ['plastic','metal','paper','cardboard'] or 'glass' in label:
        category = "♻️ Recyclable Waste"
    elif label == 'battery':
        category = "⚠️ Hazardous Waste"
    else:
        category = "🚯 Non-Recyclable Waste"

    return {
        "Prediction": label,
        "Confidence": round(confidence, 3),
        "Category": category
    }

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="♻️ Waste Classification System",
    description="Upload an image to classify waste"
)

interface.launch()