import os
from uuid import uuid4
import tensorflow
from flask import Flask, request, render_template, send_from_directory,flash
import PIL
import cv2
# from pylab import *
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model



app=Flask(__name__)
app.secret_key='random string'

classes=['Authorized','Unauthorized']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")
@app.route('/upload1/<filename>')
def send_image(filename):
    print('kjsifhuissywudhj')
    return send_from_directory("images", filename)

@app.route("/upload1", methods=["POST","GET"])
def upload1():
    print('a')
    if request.method=='POST':
        myfile = request.files['file']
        print("sdgfsdgfdf")
        fn = myfile.filename
        mypath = os.path.join('images/', fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)
        # import tensorflow as tf
        
        # img = r"D:\Fathima\Python\medical image\database\train\Eye\006.jpg"
        new_model = load_model("visualizations/model.h5")
        test_image = image.load_img(mypath, target_size=(224 ,224))
        test_image = image.img_to_array(test_image)
        test_image=test_image/255
        print(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        prediction = classes[np.argmax(result)]


        if prediction=='Authorized':
            msg='''
            Name: 
            <br /> 
            Employee ID:  
            <br />  
            Department:  
            <br /> 
            Contact no:  
            <br />
            Blood Group:  
            <br />
            Address:                

            '''
        else:
            msg='''
            Employee Not Found
            '''

        # image = cv2.imread(mypath)
        # mask = np.zeros(image.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (100, 70), (210, 180), 255, -1)
        # cv2.imshow("Rectangular Mask", mask)
        # fn1 = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Mask Applied to Image", fn1)
        # cv2.waitKey(0)
        # plt.imshow(fn1,cmap=plt.cm.gray)
    return render_template("template.html", image_name=fn,text=prediction, msg=msg)
    return render_template("upload.html")

if __name__=='__main__':
    app.run(debug=True)



# (155,100),80

