import numpy as np
import cv2

# Reading weights and cfg file for object detection model
net = cv2.dnn.readNet("yolov3_small_dataset.weights", "yolov3_tiny-training.cfg")
classes = ['stool']

# Getting information of darknet (YOLO_v3)
layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


'''MAIN OBJECT DETECTION MODEL FOR CPVISION
	img: bowel image
	return: modified image with boxes labeling stool'''
def cpVisionStoolDetection(img):

	img = cv2.resize(img, (320, 320)) 

	# Get dimensions of image
	height, width, channels = img.shape

	# Detecting obj
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(outputLayers)

	class_ids = []
	confidences = []
	boxes = []

	for out in outs:
	    for detection in out:
	        scores = detection[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]

	        # Setting a confidence threshold to greater than 50% accuracy
	        if confidence > 0.5:
	            center_x = int(detection[0] * width)
	            center_y = int(detection[1] * height)
	            w = int(detection[2] * width)
	            h = int(detection[3] * height)

	            x = int(center_x - w /2)
	            y = int(center_y - h /2)

	            boxes.append([x, y, w, h])
	            confidences.append(float(confidence))
	            class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	num_object_detected = len(boxes)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(num_object_detected):
	    if i in indexes:
	        x, y, w, h = boxes[i]
	        label = str(classes[class_ids[i]]) + ' '+ str(round(confidences[i], 2))
	        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
	        # cv2.putText(img, label, (x, y+30), font, 1, (0,0,0), 1)

	        
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   


	return img  

