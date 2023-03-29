import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions

from tensorflow.keras.applications import ResNet50
import certifi
import ssl

ssl_context = ssl.create_default_context(cafile=certifi.where())

model = ResNet50(weights='imagenet')

testimages = './testbilder'

for image_name in os.listdir(testimages):
    image_path = os.path.join(testimages, image_name)
    image = Image.open(image_path).resize((224, 224))

    x = np.array(image, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    prediction = decode_predictions(preds, top=3)[0]

    print("Prediction for", image_name, ":")
    for prediction_tuple in prediction:
        print(prediction_tuple[1], ":", prediction_tuple[2])
    print("\n")
