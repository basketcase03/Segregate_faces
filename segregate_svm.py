#https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

import cv2
import numpy as np 
import face_recognition
import os
from os import listdir
from os.path import isfile, join
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

dir_path = os.path.dirname(os.path.realpath(__file__))
trumppath = dir_path+'/trump'
testpath = dir_path+'/test'
nontrumppath = dir_path+'/not_trump'

def move_to_trump(image):
    src = testpath+'/'+image
    dest = dir_path+'/'+'segregated_trump_svm'
    shutil.copy(src, dest)

def move_to_non_trump(image):
    src = testpath+'/'+image
    dest = dir_path+'/'+'segregated_non_trump_svm'
    shutil.copy(src, dest)

def get_non_trump_encodings():
    print("Start encoding")
    non_trump_images = [f for f in listdir(nontrumppath) if isfile(nontrumppath+'/'+f)]

    non_trump_image_encodings=[]
    for image in non_trump_images:
        print("Getting encoding of image...")
        nonimgTrump = face_recognition.load_image_file(nontrumppath+'/'+image)
        nonimgTrump = cv2.cvtColor(nonimgTrump,cv2.COLOR_BGR2RGB)

        
        faceLoc = face_recognition.face_locations(nonimgTrump)
        if(len(faceLoc)>0):
            faceLoc = face_recognition.face_locations(nonimgTrump)[0]
            nonencodeTrump = face_recognition.face_encodings(nonimgTrump)[0]
            non_trump_image_encodings.append(nonencodeTrump)
    print("Encoding over")
    return non_trump_image_encodings

def get_trump_encodings():
    print("Start encoding")
    trump_images = [f for f in listdir(trumppath) if isfile(trumppath+'/'+f)]

    trump_image_encodings=[]
    for image in trump_images:
        print("Getting encoding of image...")
        imgTrump = face_recognition.load_image_file(trumppath+'/'+image)
        imgTrump = cv2.cvtColor(imgTrump,cv2.COLOR_BGR2RGB)

        
        faceLoc = face_recognition.face_locations(imgTrump)
        if(len(faceLoc)>0):
            faceLoc = face_recognition.face_locations(imgTrump)[0]
            encodeTrump = face_recognition.face_encodings(imgTrump)[0]
            trump_image_encodings.append(encodeTrump)
    print("Encoding over")
    return trump_image_encodings

def segregate(svm_model):
    test_images = [f for f in listdir(testpath) if isfile(testpath+'/'+f)]
    
    for image in test_images:
        imgTest = face_recognition.load_image_file(testpath+'/'+image)
        imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

        result = -1
        faceLoc = face_recognition.face_locations(imgTest)
        if(len(faceLoc)>0):
            faceLoc = face_recognition.face_locations(imgTest)[0]
            encodeImg = face_recognition.face_encodings(imgTest)[0]
            result = svm_model.predict([encodeImg])
            if(result==1):
                move_to_trump(image)
        if(result!=1):
            move_to_non_trump(image)
        print("image is  "+image+" result is "+str(result))

def create_dataset():
    data = []
    trump_encodings = get_trump_encodings()
    for x in trump_encodings:
        y = np.append(x,1)
        data.append(y)
    others = get_non_trump_encodings()
    for x in others:
        y = np.append(x,0)
        data.append(y)

    
    col_names = []
    for i in range(128):
        col_names.append('col'+str(i))
    col_names.append('istrump')

    df = pd.DataFrame(data, columns=col_names)
    return df

def train_svm(df):
    print("Training...")
    X = df.drop('istrump',axis=1)
    y = df['istrump']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=1234123,stratify=df['istrump'])
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    classification_report_ = classification_report(y_test, y_pred)
    print('Model accuracy is: ')
    print(classification_report_)
    svm.fit(X_train, y_train)
    return svm



def main():
    df = create_dataset()
    svm_model = train_svm(df)
    segregate(svm_model)

if __name__ == "__main__": 
    main()