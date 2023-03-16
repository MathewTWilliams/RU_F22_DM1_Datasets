# Author: Matt Williams
# Version: 10/21/2022

from pickletools import optimize
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import *
from tensorflow.keras import layers



def main(filters = 8, kernel_size = (3,3), hidden_layer_size = 16):
    num_classes = 4
    input_shape = (256,256,1)
    epochs = 5
    batch_size = 128

    x_train, y_train, x_test, y_test = get_dataset(remove_negatives=True, 
                                                    to_flatten=False, 
                                                    labels_to_int=True,
                                                    return_hist=False,
                                                    color=True)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(filters = filters, kernel_size = kernel_size, activation = "relu"), 
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Flatten(),
            layers.Dense(hidden_layer_size, activation="relu"), 
            layers.Dense(num_classes, activation = "softmax")
        ]
    )
    print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

    training_hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
        validation_data = (x_test, y_test))


    plt.plot(training_hist.history["accuracy"], label = "Accuracy")
    plt.plot(training_hist.history["val_accuracy"], label = "Validation Accuracy")
    plt.legend()
    plt.show()
    plt.clf()

if __name__ == "__main__":
    main()
    main(kernel_size=(5,5))
    main(kernel_size=(7,7))
    main(filters=4)
    main(filters=16)
    main(hidden_layer_size=8)
    main(hidden_layer_size=32)