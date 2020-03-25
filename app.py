from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# # Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer





from flask import Flask, request, redirect, url_for, flash, jsonify#
import numpy as np#
import pickle as p#
import json#








import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle






app = Flask(__name__)







# def model_predict(img_path, model):                                       //model_predict ====> defination
#     img = image.load_img(img_path, target_size=(224, 224))
#
#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)
#
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')
#
#     preds = model.predict(x)                                              // model_predict ====> call
#     return preds





 # model = (open('model.h5', 'rb'))#(open('mask_rcnn_coco.h5', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#
#     output = round(prediction[0], 2)
#
#     return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))                                               # uploads image will save here.
        f.save(file_path)

        # Make prediction
            # preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
     modelfile = 'maskrcnn.pickle'#
     model = p.load(open(modelfile, 'rb')) #
     app.run(debug=True) #app.run(debug=True)
