from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
import numpy as np
import os

from dotenv import dotenv_values

vars = dotenv_values(".env")
total = int(vars.get("train_total", 0))
optimizer =['adamw', 'rmsprop'] 

# Load trained model
# model = load_model(f'./models/{total}_model_keras.h5')
model = load_model(f'./models/{optimizer[0]}_{total}_model_keras.h5')

img_width, img_height = 150, 150
test_dir = './test_rand/'

# Load and predict
testImages = os.listdir(test_dir)

for image in testImages:
    try:
        img_path = os.path.join(test_dir, image)
        img = Image.open(img_path)
        img = img.resize((img_width, img_height))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        result = model.predict(img)
        prediction = 'dog' if result[0][0] >= 0.5 else 'cat'

        print(f"The image {image} is a: {prediction}")

    except Exception as e:
        print(f"Skipping file '{image}' due to error: {e}")
