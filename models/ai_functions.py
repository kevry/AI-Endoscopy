import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

def BBPSPredict(image, bbps_model):
  #Works if image dim = (1, x, y, z)
  #Can't work if image dim = (x, y, z)
  prob_prediction = bbps_model.predict(image)
  return np.argmax(prob_prediction)

def data_prep(im, width, height):
    #removing the botton-left green box
    im[383:560,30:248] = (0,0,0)
    #removing the botton-right logo
    im[442:482, 630:660] = (0,0,0)
    # Removing the top-left text
    im[1:50, 5:120] = (0,0,0)
    im[55:84, 30:92] = (0,0,0)
    im[80:110, 32:72] = (0,0,0)
    # Covert to RGB, downsample, and return np array
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    dsize = (int(width), int(height))
    im = cv2.resize(im, dsize)
    imnp = np.array([np.array(im)])
    return imnp

def augment_score(im, pred, bbps):
  pred_str = 'PRED: ' + str(pred)
  bbps_str = 'BBPS: ' + str(bbps)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(im, pred_str, (40,35), font, .5, (0, 255, 0), 1, cv2.LINE_AA)
  cv2.putText(im, bbps_str, (40,55), font, .5, (0, 255, 0), 1, cv2.LINE_AA)
  return im


im = cv2.imread('bowel.png', 1)
pred = 1
bbps = 2
# augment_score(im, pred, bbps)