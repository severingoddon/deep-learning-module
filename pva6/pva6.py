from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


def get_img_array(img_path, target_size):
    img = keras.utils.load_img(
        img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array
img_tensor = get_img_array("test.jpg", target_size=(180, 180))



# Define input shape
input_shape = (180, 180, 3)

# Define input layer
input_layer = Input(shape=input_shape)

# Add convolutional layers
conv_layer1 = Conv2D(32, kernel_size=(4,4), activation='relu')(input_layer)
pool_layer1 = MaxPooling2D(pool_size=(2,2))(conv_layer1)
conv_layer2 = Conv2D(64, kernel_size=(3,3), activation='relu')(pool_layer1)
pool_layer2 = MaxPooling2D(pool_size=(2,2))(conv_layer2)
conv_layer3 = Conv2D(128, kernel_size=(3,3), activation='relu')(pool_layer2)
pool_layer3 = MaxPooling2D(pool_size=(2,2))(conv_layer3)
conv_layer4 = Conv2D(256, kernel_size=(3,3), activation='relu')(pool_layer3)

# Add flatten layer
flatten_layer = Flatten()(conv_layer4)

# Add dense layers
dense_layer1 = Dense(512, activation='relu')(flatten_layer)
dense_layer2 = Dense(10, activation='softmax')(dense_layer1)

# Define model input and output
model = Model(inputs=input_layer, outputs=dense_layer2)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

layer_outputs = []
layer_names = []
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape) 

plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")