import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

out_labels = open("labels.txt", "r")
out_sequences = open("sequences.txt", "r")
out_classes = open("classes.txt", "r")

longest = 0

labels = []
for i in out_labels:
    labels.append(int(i.strip()))

sequences = []
for s in out_sequences:
    if (longest < len(s.strip())):
        longest = len(s.strip())
    sequences.append([int(x) for x in s.strip()])

classes = []
for i in out_classes:
    classes.append(i.strip())

samples = len(labels)
clas = len(classes)

out_labels.close()
out_sequences.close()
out_classes.close()

sequences = sequence.pad_sequences(sequences, maxlen=longest)

X = np.array(sequences)
y = np.array(labels)

y = utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
np.random.seed(42)
X_train=np.random.permutation(X_train)
np.random.seed(42)
y_train=np.random.permutation(y_train)

layers = [1, 2, 5]
alpha = [0.001, 0.0001, 0.00001]
dropout = [0.1, 0.5, 0.7]
epoch = [10, 20]

best = {"a": 0, "d": 0, "e": 0, "val": 0, "test": 0}
for d in dropout:
    for e in epoch:
        for a in alpha:
            for l in layers:
                model = Sequential()
                model.add(Embedding(input_dim=5, output_dim=clas, input_length=longest))
                model.add(Dropout(d))
                for i in range(l - 1):
                    model.add(LSTM(units=64, dropout=d, recurrent_dropout=d, return_sequences=True))
                model.add(LSTM(units=64, dropout=d, recurrent_dropout=d))
                model.add(Dense(clas, activation='softmax'))

                model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(a), metrics=['accuracy'])

                history = model.fit(X_train, y_train, batch_size=64, validation_split=0.1, epochs=e)
                scores = model.evaluate(X_test, y_test, verbose=0)
                print("Accuracy: %.2f%%" % (scores[1] * 100))

                if (best["val"] < history.history['val_accuracy'][-1]):
                    best["a"] = a
                    best["e"] = e
                    best["d"] = d
                    best["val"] = history.history['val_accuracy'][-1]
                    best["test"] = scores[1]

                fig = plt.figure()
                plt.plot(history.history['loss'], label='training loss')
                plt.plot(history.history['val_loss'], label='validation loss')
                plt.legend(loc='best')
                plt.title("Training and validation loss")
                fig.savefig(
                    "./pics/Loss_{alpha}_{dropout}_{epoch}.jpg".format(alpha=a, dropout=d, epoch=e))
                plt.close(fig)

                fig = plt.figure()
                plt.plot(history.history['accuracy'], label='train accuracy')
                plt.plot(history.history['val_accuracy'], label='validation accuracy')
                plt.legend(loc='best')
                plt.title("Training and validation accuracy")
                fig.savefig(
                    "./pics/Accuracy_{alpha}_{dropout}_{epoch}.jpg".format(alpha=a, dropout=d, epoch=e))
                plt.close(fig)

out_train = open("result.txt", "w")
out_train.write("Best validation: " + str(best["val"]) + "\n")
out_train.write("Best test: " + str(best["test"]) + "\n")
out_train.write("Best epochs: " + str(best["e"]) + "\n")
out_train.write("Best dropout: " + str(best["d"]) + "\n")
out_train.write("Best alpha: " + str(best["a"]) + "\n")
out_train.close()
