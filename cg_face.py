
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
print(tf.__version__)

'''

 yolo for face swap
 
 note:
    Not all media player can decode our saved video.
    Tested successfully on:
        HUplayer, nplayer(ios)
    Failed on:
        potplayer, windows player

'''

#Configs

MX=10   #   output matrix width
MY=7    #   output matrix height
MW=320  #   input matrix width
MH=240  #   input matrix height


MODEL='./yolo.h5'           #yolo model
FACE_PIC='./faf.jpg'        #the picture you wangtu replace the real face
VIDEO_SRC='./lie.mp4'       #source video
VIDEO_SAVE='./save.mp4'     #save video(all processed frames)
VIDEO_SAMPLE='./sample.mp4' #save sample(frames that contain bboxs only)


#trained on celbea dataset
#yolo_loss lite version
'''no use in this script'''
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


#init
model = load_model(MODEL,custom_objects={'loss function': loss_func,'loss_func':loss_func})
video = cv2.VideoCapture(VIDEO_SRC)#open src video
cg=cv2.imread(FACE_PIC)#set up face-swap pic

#get video info 
fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
WIDTH=size[0]
HEIGHT=size[1]

print(fps,size)

#set up video writer
video_writer = cv2.VideoWriter('lie yww2.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
frame_writer = cv2.VideoWriter('sample lie2.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

#set up start frame  
video.set(cv2.cv2.CAP_PROP_POS_FRAMES,8000)#141500

while  video.isOpened():            #use this if you want to process whole video
#for i in range(0,200):             #use this if you want to process specific frames

    if video.grab() and cv2.waitKey(1) & 0xFF != ord('q'):
        success, frame = video.retrieve()
    else:
        break
        
    img=cv2.resize(frame, (MW, MH),)
    #convert bgr->rgb
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = np.array([img])
    #get predict result
    pred = model.predict(img)[0]
    face=[]
    for y in range(0,MY):
            for x in range(0,MX):
                if(pred[y][x][0]>0.9):
                    print(x,y,pred[y][x])
                    '''
                    (bx1,by1)-------------
                    |                    |
                    |        bbox        |
                    |                    |
                    |------------(bx2,by2)
                    '''
                    bx1=int(pred[y][x][1]*WIDTH/MX)
                    by1=int(pred[y][x][2]*HEIGHT/MY)
                    bx2=int(pred[y][x][3]*WIDTH)
                    by2=int(pred[y][x][4]*HEIGHT)
                    face=frame[by1:by1+by2,bx1:bx1+bx2,:]
                    try:
                        cgf=cv2.resize(cg, (bx2, by2),)
                        cv2.rectangle(frame,(bx1,by1),(bx1+bx2,by1+by2),(0,255,0),5)
                        frame[by1:(by1+by2),bx1:(bx1+bx2),:]=cgf[:][:][:]
                        frame_writer.write(frame)
                    except:
                        cv2.rectangle(frame,(bx1,by1),(bx1+bx2,by1+by2),(0,255,0),5)
    cv2.putText(frame,str(i),(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    #convert rgb->bgr
    r,g,b = cv2.split(img[0])
    
    img = cv2.merge([b,g,r])
    img = np.array([img])
    video_writer.write(frame)
    cv2.imshow('face',frame)
    
cv2.destroyAllWindows()
video.release()
video_writer.release()
frame_writer.release()
#model = load_model('D:/Desktop/g_design/yolo/',custom_objects={'loss function': loss_func,'loss_func':loss_func})
