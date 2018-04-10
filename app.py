# USAGE
# Start the server:
# 	python app.py

# import the necessary packages
import numpy as np
import flask
from flask import request
import io
import json
#np.set_printoptions(threshold=np.inf)
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

# we're still going to use a Tokenizer here, but we don't need to fit it
max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
# for human-friendly printing
labels = ['HALB', 'ZPAK', 'ZRAW']

# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
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


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "GET":
		str = request.args.get('material')
		testArr = convert_text_to_index_array(str)
		input = tokenize.sequences_to_matrix([testArr], mode='binary')
		pred = model.predict(input)
		print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))

		data["analysis"] = []
		r = {"label": labels[np.argmax(pred)], "probability": float(pred[0][np.argmax(pred)] * 100)}
		data["analysis"].append(r)
		data["success"] = True

		data["label"] = labels[np.argmax(pred)]
		data["probability"] = float(pred[0][np.argmax(pred)] * 100)

		data["HALB"] = float(pred[0][0] * 100)
		data["ZPAK"] = float(pred[0][1] * 100)
		data["ZRAW"] = float(pred[0][2] * 100)
			

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()
