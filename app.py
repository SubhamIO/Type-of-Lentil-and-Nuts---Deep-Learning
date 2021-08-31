# Importing essential libraries
import tensorflow as tf
from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import numpy as np
import pathlib, os, json, pickle

import pandas as pd
from PIL import Image
#from utility_functions import *

#======Utility Fuction========
def classifier(img_file):
    img_temp = Image.open(img_file)
    img_temp = img_temp.convert('RGB')
    img_temp.save('./testing/test1/1.jpg')
    # with open(os.path.join('testing/test1/','1.jpg'), 'wb') as file:
        # file.write(img_file.getbuffer())

    # ***================= Loading the model =================***
    model = tf.keras.models.load_model('checkpoint/weightings.h5') # load the model structure and weight
    with open(os.path.join('checkpoint','label_to_index.txt'),'r') as file: # load the trained labels
        label_names = json.loads(file.read())
    label_names = {y:x for x, y in label_names.items()} # transform from label:number to number:label


    #%%
    # ***================= Classify the whisky in testing folder =================***

    #============ Creating image file reader ============
    def preprocess_image(image): # Input an image, resize it to 224, normalize the data from 0-255 to 0-1
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.crop_to_bounding_box(image,int(image.shape[0]*0.2),0,int(image.shape[0]*0.6),int(image.shape[1])) # Cutting the left 20% and right 20% out
        # plt.imshow(image.numpy(), cmap='gray') # Print the cut image for testing
        # plt.show()
        image = tf.image.resize(image, [224, 224])
        image /= 255.0
        return image

    def load_and_preprocess_image(path): # Input a path, read the image, and preprocess it
        image = tf.io.read_file(path)
        return preprocess_image(image)

    #============ Classifying the lentil ============
    testing_root = pathlib.Path('testing/')
    all_testing_paths = [str(path) for path in list(testing_root.glob('*/*'))]

    pred_list = []
    for i in range(len(all_testing_paths)):
        img = load_and_preprocess_image(all_testing_paths[i])
        img = np.reshape(img, [1,224,224,3])
        pred_list.append(label_names[np.argmax(model.predict(img))])
        #test = enumerate(model.predict(img).tolist()[0])



    return str(pred_list[-1])



#==============Loading the model========

model = tf.keras.models.load_model('checkpoint/weightings.h5')#Load model structure and weightings
with open(os.path.join('checkpoint','label_to_index.txt'),'r') as file: #load the trained label_names
    label_names = json.loads(file.read())
label_names= {y:x for x,y in label_names.items()}#transform from label:number to number:label


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['my_image']
        # Read the image via file.stream
        index = classifier(img_file.stream)

    #     return jsonify({"Lentil Type": index})
    # else:
    #     return jsonify({"Lentil Type": 'N/A'})

    return render_template('result.html', my_prediction = index)



if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 8080,debug=True)
	#app.run(debug=True)
