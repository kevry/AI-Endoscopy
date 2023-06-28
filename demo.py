import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from ai_functions import BBPSPredict, data_prep, augment_score
import time

full_path = "models/model_256_81"
bbps_model = tf.keras.models.load_model(full_path)

inputpath = 'videos/class2_60fps.mp4'
cap = cv2.VideoCapture(inputpath)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20
bbps = 2 # BBPS

out = cv2.VideoWriter('predictions/class0_predictor.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))

count = 0
ccr = 0
num = 0
duration = 0
while(cap.isOpened()):
    start = time.time()
    ret, frame = cap.read()
    fr = cap.get(1)
    if ret:
        if count == 2:
            num+=1
            model_input = data_prep(frame, 256, 256) # model input is np array
            pred = BBPSPredict(model_input, bbps_model) # pred is string scalar
            # if pred == bbps:
            #     ccr+=1
            vidout = augment_score(frame, pred, bbps) # frame is a video frame
            out.write(vidout) #write frames of vidout function
            count = 0
            cv2.imshow('Real-Time Predicions',frame)
        else:
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    stop = time.time()
    duration = duration + (stop-start)
    print(stop-start)
# ccr = round(ccr/num, 4)
print(duration)
print(num)
avg = duration/num
print('%.10f' % avg)
print("\n\nThe Correct Classification Rate (CCR) for this video is: " + str(ccr))
print("The video output is 20FPS")
cap.release()
out.release()
cv2.destroyAllWindows()
