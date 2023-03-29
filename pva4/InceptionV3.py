import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
import certifi
import ssl

ssl_context = ssl.create_default_context(cafile=certifi.where())
model = InceptionV3(weights='imagenet')
testimages = './testbilder'

for image_name in os.listdir(testimages):
    image_path = os.path.join(testimages, image_name)
    image = Image.open(image_path).resize((299, 299))

    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    predictions = decode_predictions(preds, top=3)[0]

    print("Predictions for", image_name, ":")
    for prediction in predictions:
        print(prediction[1], ":", prediction[2])
    print("\n")
