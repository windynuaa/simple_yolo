

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

print(tf.__version__)

#yolo for face

#trained on celbera dataset


#yolo_loss lite version
def loss_func(y_pred,y_true):
    loss=tf.reduce_sum(tf.square(y_pred[...,1:5]-y_true[...,1:5]))   * 5  #location loss
    loss_class=tf.reduce_sum(tf.square(y_pred[...,0]-y_true[...,0])) * 1  #class loss
    return loss+loss_class
    
#image input w:320 h:240
'''
output structure:

        c 1                 c 2         ...        c 10
---------------------------------------------------------------- 
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 1
----------------------------------------------------------------
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 2  
----------------------------------------------------------------
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 3
----------------------------------------------------------------
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 4
----------------------------------------------------------------
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 5
----------------------------------------------------------------
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 6
----------------------------------------------------------------
(confident,x,y,w,h)|(confident,x,y,w,h)|...|(confident,x,y,w,h)|  row 7
---------------------------------------------------------------
'''
#net output 7(grd_y)*10(grd_x)*5(confident,x,y,w,h)
model = load_model(r'.\yolo.h5',custom_objects={'loss function': loss_func,'loss_func':loss_func})

capture = cv2.VideoCapture(0)

while 1:
    ret, frame = capture.read()
    img=cv2.resize(frame, (320, 240),)
    #convert bgr->rgb
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = np.array([img])
    #get predict result
    pred = model.predict(img)[0]

    for y in range(0,7):
            for x in range(0,10):
                if(pred[y][x][0]>0.8):
                    print(x,y,pred[y][x])
                    '''
                    (bx1,by1)-------------
                    |                    |
                    |        bbox        |
                    |                    |
                    |------------(bx2,by2)
                    '''
                    bx1=int(pred[y][x][1]*320/10)
                    by1=int(pred[y][x][2]*240/7)
                    bx2=int(pred[y][x][3]*320)
                    by2=int(pred[y][x][4]*240)
                    cv2.rectangle(img[0],(bx1,by1),(bx1+bx2,by1+by2),(255,0,0),2)
                    
    #convert rgb->bgr
    r,g,b = cv2.split(img[0])
    img = cv2.merge([b,g,r])
    img = np.array([img])
    cv2.imshow('face',img[0])
    cv2.waitKey(1)
capture.release()


#model = load_model('D:/Desktop/g_design/yolo/',custom_objects={'loss function': loss_func,'loss_func':loss_func})