import cv2 as cv
import imutils
import time
import numpy as np
model_face_bin = 'fastsal_cocoA.bin'
model_face_xml = 'fastsal_cocoA.xml'

net = cv.dnn.readNet(model_face_xml,model_face_bin)
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

capture = cv.VideoCapture(0)
ret, frame = capture.read()
frame = imutils.resize(frame, height=192, width=256)
blob = cv.dnn.blobFromImage(frame,size=(frame.shape[1],frame.shape[0]),ddepth=cv.CV_8U)
net.setInput(blob)
net.forward()

speeds = []
for i in range(1000):
        blob = cv.dnn.blobFromImage(frame,size=(frame.shape[1],frame.shape[0]),ddepth=cv.CV_8U)
        net.setInput(blob)
        start = time.time()
        net.forward()
        interval=(time.time()-start)
        speeds.append((time.time() - start) * 1000)
        print('%.2fms|%.2fms|%.2fms|%.2f' % (np.min(speeds), np.median(speeds), np.mean(speeds), np.std(speeds)))
        fps=1/interval
        print("Interval="+str(interval))
        print("FPS="+str(fps))
