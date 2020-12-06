#all imports here
from flask import Flask, render_template, request, redirect, flash, url_for
import numpy as np
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import model_from_json
import cv2


#categories of facial expression
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

#folder were all the images are stored which are used for prediction
UPLOAD_FOLDER = 'C:/Users/abhis/OneDrive/Desktop/Project/FML_Submission/photos'

#Haar Cascade library of openCV to detect faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#constructor for flask
app = Flask(__name__)
app.secret_key = "secret key"
#configuring the upload folder to store images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#class for making predictions
class Prediction_Maker(object):
    def __init__(self, model_json, model_weights):
        #loading the json file storing the CNN model
        with open(model_json, "r") as json_file:
            json_content = json_file.read()
            #parse the json model configuration and return a model
            self.model = model_from_json(json_content)
        
        #Loads all layer weights, either from a TensorFlow or an HDF5 weight file.
        self.model.load_weights(model_weights)
        #Creates a function that executes one step of inference.
        self.model.make_predict_function()
        #size of the images
        self.IMG_SIZE = 48
    
    #detect the face and convert to grascale
    def prepare_image(self, img):
        #read the image and convert to grayscale
        img_gray_arry = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        #use the haar cascade model to detect the face in the image
        face = face_detector.detectMultiScale(img_gray_arry, 1.3, 5)
        #use the face coordinate(x, y), height and width to separate face from rest of the picture
        for (x, y, w, h) in face:
            img_gray_arry = img_gray_arry[y:y+h, x:x+w]
        
        #resize the image to 48x48 to fit the model 
        new_arry = cv2.resize(img_gray_arry, (self.IMG_SIZE, self.IMG_SIZE))
        #convert to (48x48x1) and return it to model
        return new_arry.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

    #use the model predict function to get a vector of size 7 of softmax values
    def make_prediction(self, img):
        return self.model.predict(self.prepare_image(img))

#object of the model
model = Prediction_Maker("model.json", "model_weights.h5")

#start the web app
@app.route('/')
def index():
    return render_template('index.html')


#upload file commands
#reference: https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            #save the image in photos folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #read the image and make prediction
            label = model.make_prediction("photos/"+filename)
            #return the emotion to index.html
            flash(CATEGORIES[np.argmax(label)])
            
            return redirect('/')

if __name__ == "__main__":
    app.run()