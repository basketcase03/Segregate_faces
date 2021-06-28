#https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

import cv2
import numpy as np 
import face_recognition
import os
from os import listdir
from os.path import isfile, join
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
trumppath = dir_path+'/trump'
testpath = dir_path+'/test'

def move_to_trump(image):
    src = testpath+'/'+image
    dest = dir_path+'/'+'segregated_trump'
    shutil.copy(src, dest)

def move_to_non_trump(image):
    src = testpath+'/'+image
    dest = dir_path+'/'+'segregated_non_trump'
    shutil.copy(src, dest)


def get_trump_encodings():
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
    return trump_image_encodings

def segregate():
    test_images = [f for f in listdir(testpath) if isfile(testpath+'/'+f)]
    
    trump_image_encodings = get_trump_encodings()
    for image in test_images:
        imgTest = face_recognition.load_image_file(testpath+'/'+image)
        imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

        result = -1
        faceLoc = face_recognition.face_locations(imgTest)
        if(len(faceLoc)>0):
            faceLoc = face_recognition.face_locations(imgTest)[0]
            encodeImg = face_recognition.face_encodings(imgTest)[0]
            results = face_recognition.compare_faces(trump_image_encodings,encodeImg)
            trues = sum(bool(x) for x in results)
            falses = len(results) - trues
            result = trues - falses
            if(result>=0):
                move_to_trump(image)
        if(result<0):
            move_to_non_trump(image)
        print("image is  "+image+" result is "+str(result))



def main():
    segregate()

if __name__ == "__main__": 
    main()