import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from ai_functions import BBPSPredict, data_prep, augment_score, postAnalysis
import time
from matplotlib import pyplot as plt
from datetime import datetime, date

full_path = "models/model_256_81"
bbps_model = tf.keras.models.load_model(full_path)

inputpath = 'videos/class2301.mp4'
cap = cv2.VideoCapture(inputpath)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

## Running average code
count = 0
runsum = 0;

## Begin camera capture
track_list = []
track_time = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        model_input = data_prep(frame, 256, 256) # model input is np array
        if (count==0 or count%5==0):
            start = time.time()
            pred = BBPSPredict(model_input, bbps_model) # pred is string scalar
            now = datetime.now()
            track_time.append(now)
            track_list.append(pred)
            runsum += pred;
            runavg = round(runsum/(count/5+1),2);
            stop = time.time()
        augment_score(frame, pred, runavg) # frame is a video frame  
        cv2.imshow('Real-Time Predicions',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    count += 1

postAnalysis(track_time, track_list) # Plotting
cap.release()
cv2.destroyAllWindows()
