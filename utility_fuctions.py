import tensorflow as tf
import numpy as np
import pathlib, os, json, pickle
# import IPython.display as display
# import matplotlib.pyplot as plt
import pandas as pd
# import streamlit as st
from PIL import Image


#%%
# ***================= Steamlit image upload =================***
# img_file = st.file_uploader('Upload a whisky image')
# if img_file != None:
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
