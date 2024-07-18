from fastapi import FastAPI, UploadFile, File
import json
from PIL import Image
from io import BytesIO
import numpy as np
from model import build_model
import tensorflow as tf

app = FastAPI()

# Load model
model = build_model()
model.load_weights('./model/gen_25.weights.h5')
classes = ['Ahegao','Angry','Happy','Neutral','Sad','Surprise']

@app.get("/")
def first_api():
    return {
        "response": "Face Expression Image Generation"
    }

@app.post("/generate")
async def generate(label):
    label_index = classes.index(label)
    if(label_index >= 0 && label_index <= len(classes)):
        num_examples = 1
        latent_dim = 100
        noise = tf.random.normal([num_examples, latent_dim])
        labels = tf.repeat(label_index, [1], axis=None, name=None)
        predictions = model([noise, labels], training=False)
        pred = (predictions[0, :, :, :] + 1 ) * 127.5
        pred = np.array(pred)
        return {
            "label": label,
            "Img": pred
        }
    else:
        return {
            "label": "Invalid label:"+label,
            "Img": []
        }