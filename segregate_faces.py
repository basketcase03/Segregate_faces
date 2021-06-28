import cv2
import numpy as np 
import face_recognition
import os
from os import listdir
from os.path import isfile, join
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

dir_path = os.path.dirname(os.path.realpath(__file__))
tosegregatepath = dir_path+'/trump'   #Enter path to your directory of images
testpath = dir_path+'/test'           #Enter path to which images you want to segregate
otherspath = dir_path+'/not_trump'      #Enter path to dir which contain images that are not you

def move_to_segregated(image):
    src = testpath+'/'+image
    dest = dir_path+'/'+'segregated_trump_multiface'    #Enter path where you want to keep images
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.copy(src, dest)

def moce_to_others(image):
    src = testpath+'/'+image
    dest = dir_path+'/'+'segregated_ntrump_multiface'     #Enter path where you want to keep images not yours
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.copy(src, dest)

def get_others_encoding():
    print("Start encoding")
    others_images = [f for f in listdir(otherspath) if isfile(otherspath+'/'+f)]

    others_encodings=[]
    for image in others_images:
        print("Getting encoding of image...")
        othersimg= face_recognition.load_image_file(otherspath+'/'+image)
        othersimg= cv2.cvtColor(othersimg,cv2.COLOR_BGR2RGB)

        faceLoc = face_recognition.face_locations(othersimg)
        if(len(faceLoc)>0):
            faceLoc = face_recognition.face_locations(othersimg)[0]
            othersencode = face_recognition.face_encodings(othersimg)[0]
            others_encodings.append(othersencode)
    print("Encoding over")
    return others_encodings

def get_segregate_encodings():
    print("Start encoding")
    segregat_images = [f for f in listdir(tosegregatepath) if isfile(tosegregatepath+'/'+f)]

    to_Segregate_encodings=[]
    for image in segregat_images:
        print("Getting encoding of image...")
        imgsegregate = face_recognition.load_image_file(tosegregatepath+'/'+image)
        imgsegregate = cv2.cvtColor(imgsegregate,cv2.COLOR_BGR2RGB)

        
        faceLoc = face_recognition.face_locations(imgsegregate)
        if(len(faceLoc)>0):
            faceLoc = face_recognition.face_locations(imgsegregate)[0]
            segregate_encode = face_recognition.face_encodings(imgsegregate)[0]
            to_Segregate_encodings.append(segregate_encode)
    print("Encoding over")
    return to_Segregate_encodings

def segregate(svm_model):
    test_images = [f for f in listdir(testpath) if isfile(testpath+'/'+f)]
    
    for image in test_images:
        imgTest = face_recognition.load_image_file(testpath+'/'+image)
        imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

        result = -1
        faceLoc = face_recognition.face_locations(imgTest)
        if(len(faceLoc)>0):
            for i in range(0,len(faceLoc)):
                faceLoc = face_recognition.face_locations(imgTest)[i]
                encodeImg = face_recognition.face_encodings(imgTest)[i]
                result = svm_model.predict([encodeImg])
                if(result==1):
                    break
        if(result!=1):
            moce_to_others(image)
        else:
         move_to_segregated(image)
        print("image is  "+image+" result is "+str(result))

def create_dataset():
    data = []
    to_segregate_encode = get_segregate_encodings()
    for x in to_segregate_encode:
        y = np.append(x,1)
        data.append(y)
    others = get_others_encoding()
    for x in others:
        y = np.append(x,0)
        data.append(y)

    
    col_names = []
    for i in range(128):
        col_names.append('col'+str(i))
    col_names.append('isSamePerson')

    df = pd.DataFrame(data, columns=col_names)
    return df

def train_svm(df):
    print("Training...")
    X = df.drop('isSamePerson',axis=1)
    y = df['isSamePerson']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=1234123,stratify=df['isSamePerson'])
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