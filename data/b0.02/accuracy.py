import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_accuracy(sigma_w, L):
    sigma_b = 0.02
    N = 8
    
    w_init = keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_w/N))
    w_init_first = keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_w/(28*28)))
    b_init = keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(sigma_b))

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten())
    for i in range(L):
        if i == 0:
            model.add(layers.Dense(N, activation='tanh', kernel_initializer=w_init_first, bias_initializer=b_init))
        else:
            model.add(layers.Dense(N, activation='tanh', kernel_initializer=w_init, bias_initializer=b_init))

    model.add(layers.Dense(num_classes, activation='softmax', kernel_initializer=w_init, bias_initializer=b_init))

    batch_size = 64
    epochs = 100

    optimizer = keras.optimizers.SGD(learning_rate=1e-3, momentum=0.8)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    sigma_w_txt = '{:.3f}'.format(sigma_w)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/Model{epoch:08d}-'+sigma_w_txt+'.h5',
        save_weights_only=False,
        save_freq=5*(x_train.shape[0] // batch_size))

    term = tf.keras.callbacks.TerminateOnNaN()
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, min_delta=0.002)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[term, model_checkpoint_callback])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score, history.history

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

sws = np.linspace(0.1, 100, 100)

score = []

for x in sws:
    score += [x, 1, *get_accuracy(x, 1)]

score = np.array(score)
np.save('accuracySlice.npy', score)
