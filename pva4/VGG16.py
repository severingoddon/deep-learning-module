import os
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import certifi
import ssl

ssl_context = ssl.create_default_context(cafile=certifi.where())

model = VGG16(weights='imagenet')
testimages = './testbilder'

for bild_name in os.listdir(testimages):
    image_path = os.path.join(testimages, bild_name)
    bild = Image.open(image_path).resize((224, 224))

    x = np.array(bild, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    vorhersage = decode_predictions(preds, top=3)[0]

    print("Vorhersage für", bild_name, ":")
    for vorhersage_tuple in vorhersage:
        print(vorhersage_tuple[1], ":", vorhersage_tuple[2])
    print("\n")
