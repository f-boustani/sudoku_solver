import cv2
import numpy as np
from datetime import datetime

def rectify(h):
        h=np.vstack(h)
        hnew = np.zeros((4,2),dtype = np.float32)
 
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
         
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
  
        return hnew

#####################################################


def image_processing(image):
    
    img =  cv2.imread(image)


    ''' threshholding the photo '''
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
   # cv2.imwrite('threshhold.jpg',thresh)
   # cv2.imshow('thresh',thresh)
   # cv2.waitKey(0)


    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    biggest = None
    max_area = 0
    for i in contours:
            area = cv2.contourArea(i)
            if area > 100:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area

    

    biggest=rectify(biggest)
    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
    

    ''' remap the photo '''
    
    retval = cv2.getPerspectiveTransform(biggest,h)
    warp = cv2.warpPerspective(gray,retval,(450,450))
    out = warp.copy()
    final = cv2.warpPerspective(img,retval,(450,450))
  #  cv2.imwrite('remap.jpg',final)
  #  cv2.imshow('remap',warp)
  #  cv2.waitKey(0)
    



    ''' find the squares and numbers '''
    
    thresh = cv2.adaptiveThreshold(warp,255,1,1,11,2)
    digits=['0' for i in range(81)]
    position = []
    model = train()
    W=450
    H=450
    d=dict()
    pos=0
    name = 0
    for i in range(9):
        for j in range(9):

            h_start = i*(H/9)
            h_end = (i+1)*(H/9)
            w_start = j*(W/9)
            w_end = (j+1)*(W/9)

            
            gri = warp[h_start:h_end,w_start:w_end]
            out = gri.copy()
            
            im = thresh[h_start:h_end,w_start:w_end]
            contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            t=0
            for c in contours:
                if cv2.contourArea(c)>50:
                    [x,y,w,h] = cv2.boundingRect(c)

                    if  not((25<h<40) and (8<w<30)):
                        continue

                    if  (25<h<40) and (8<w<30):
                        cv2.rectangle(gri,(x,y),(x+w,y+h),(0,0,255),2)
                        
                        n = str(name)+'.jpg'
                        d[n]  = pos
                        output = out[y:y+h,x:x+w]
                        cv2.imwrite(n,output)
                        position.append(0)
                        t = 1
                        name+=1
                        pos +=1
                        
                    
                        

            if(t==0):
                position.append(((w_start)+ 15 , (h_start)+ 42))
                pos +=1
                
    
    board = recognizer(name,d,digits)
   # print board
   # cv2.imwrite('recognize.jpg',warp)
   # cv2.imshow('final',warp)
   # cv2.waitKey(0)
    return board,position,final




########################## recognizer #################################################

def recognizer(name,d,digits):
        
        ''' recognize the digits in photo '''
        
        model = train()
        for i in range(name):
            n=str(i)+'.jpg'
            pos=d[n]
            img =  cv2.imread(n)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
            vector = cv2.resize(thresh,(10,10))
            vector = vector.reshape((1,100))
            vector = np.float32(vector)
            retval, results, neigh_resp, dists = model.find_nearest(vector, k = 1)
            digits[pos]= str(int(results[0][0]))
                        
        
        return digits
    
###########################prepare data for training #############################

def prepare_data():

    ''' prepare data for training '''
    
    samples=np.empty((0,100))
    responses =[]
    for j in range(1,10):
            for i in range(1,10):

                    n= str(j)+str(i)+'.jpg'
                    img = cv2.imread(n)
                    im3 = img.copy()

                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    
                    blur = cv2.GaussianBlur(gray,(5,5),0)
                    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
                    vector = cv2.resize(thresh,(10,10))
                    sample = vector.reshape((1,100))
                    samples = np.append(samples,sample,0)
                    responses.append(j)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((1,responses.size))
    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
            
##########################################################
def train():

    ''' tarin the neural network '''
    
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((1,responses.size))
    
    model = cv2.KNearest()
    model.train(samples,responses)
    return model



#image_processing('t.jpg')
