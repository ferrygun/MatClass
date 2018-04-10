from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras.models import model_from_json
import keras.preprocessing.text as kpt

# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')

evalSentence = input('Input a sentence to be evaluated, or Enter to quit: ')
testArr = convert_text_to_index_array(evalSentence)
#print(testArr)

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
input = tokenize.sequences_to_matrix([testArr], mode='binary')

labels = ['HALB', 'ZPAK', 'ZRAW']

pred = model.predict(input)
print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))

print("HALB ",  pred[0][0])
print("ZPAK",  pred[0][1])
print("ZRAW",  pred[0][2])

