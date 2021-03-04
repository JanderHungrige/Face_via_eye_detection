#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:09:14 2021

@author: Jan Werth / base
"""

import numpy as np
import cv2
import matplotlib as plt

cascaderpath='./cascader/'
SF=1.1
MN=4

"""
 Load face cascader
"""
cascader=['haarcascade_frontalface_alt',
          'haarcascade_frontalface_alt2', 
          'lbpcascade_frontalface',
          'lbpcascade_frontalface_improved',
          'haarcascade_eye',
          'haarcascade_eye_tree_eyeglasses']
face_cascade = cv2.CascadeClassifier(cascaderpath + cascader[4] + '.xml')
print('cascader loaded  ...')

"""
Define functions'
"""

def draw_rect(x,y,w,h,frame,*n,**rest):
    #face=frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = str(n[0])
    cv2.putText(frame, label, (x, y-4),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #draw index to eye
    if bool(rest)==True:
        cv2.rectangle(frame, (x-w, y-3*h), (x+2*w, y+4*h), (255, 255, 255), 2) #Draw face from eye
    else:
        cv2.rectangle(frame, (x-w, y-3*h), (x+4*w, y+4*h), (255, 255, 255), 2) #Draw face from eye
        
    cv2.imshow('frame', img) 

    return frame

def eye_detetion(img,SF,MN):
        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=SF, minNeighbors=MN)#, Size(50,50))
        if len(faces_detected) !=0:
            n=-1
           # ONE EYE              
            if len(faces_detected) == 1: # one eye means one face
                (x,y,w,h)=faces_detected
                frame=draw_rect(x,y,w,h,img)
                
            # TWO EYES   
            elif len(faces_detected) ==2 : #Detecting two eyes max
                (x1,y1,w1,h1)=faces_detected[0]
                (x2,y2,w2,h2)=faces_detected[1]
                if x1-x2 < 2*w1: # Eyes of the same face
                    (x,y,w,h)=(x2-x1,y2-y1,w2-w1,h2,h1) #draw inbetween both
                    frame=draw_rect(x,y,w,h,img)
                elif x1-x2 > 2*w1: # each Eyes of differnet faces    
                    for (x,y,w,h) in faces_detected:
                        frame=draw_rect(x,y,w,h,img) # just draw each eye            
                
            # THREE EYES +      
            elif len(faces_detected) > 2:    # more than to eyes is more than two faces 
                Dist=[]
                doubles=[]
                Dist_check=[]
                for i in range(len(faces_detected)): # go through all eyes
                    doubles.append(i) 
                    for j in range(len(faces_detected)):# compare them to all eye except themsleves
                        if j not in doubles: #prohibit doubles
                            Eye=np.array(faces_detected[i,:])
                            Eye2=np.array(faces_detected[j,:])
                            Dist.append([np.linalg.norm(Eye2-Eye),i,j]) # Get euclidian distance
                        Dist.sort()      # sort from shortest to longest distance between found eyes    
                            
                for o in range(len(Dist)): # write sorted disance list into other list, but each eye appears only once
                    print(o)
                    if (Dist[o][1] or Dist[o][2]) in [item for sub_list in Dist_check for item in sub_list]:
                            print(str(o) +' already in')
                            continue 
                    else:    
                        Dist_check.append(Dist[o][1:3])
                        print(str(o) +' added')
                # now we have found the matching eye pairs. If uneven nr of eyes, one is left out.
                leftover = np.setdiff1d(doubles,Dist_check) #Find the leftover eye
                #Now put from each pair the left eye and the leftovers into a final list
                for i in range(len(Dist_check)):
                               if faces_detected[Dist_check[i][0]][0]>faces_detected[Dist_check[i][1]][0]: #if the first of the pair of eyes has a lower x value, it is more to the right and is keept
                                   Dist_check[i].pop(0)
                               else:
                                   Dist_check[i].pop(1)
                #Dist_check.append(leftover)  # add te leftover to the list
                flat_list = [item for sublist in Dist_check for item in sublist] # flatten the list
                for (x,y,w,h) in faces_detected[flat_list]:
                    n=n+1  
                    frame=draw_rect(x,y,w,h,img,n) # just draw each eye       
                for (x,y,w,h) in faces_detected[leftover]:
                    n=n+1  
                    frame=draw_rect(x,y,w,h,img,n,rest=True) # just draw each eye for the leftover  
                    
            return frame
            
def save_image(name, frame, scaleFactor,minNeighbors):
    cv2.imwrite('./' + name + str(scaleFactor) + str(minNeighbors) + '.jpeg', frame) #save image
   # plt.savefig('./' + name + str(scaleFactor) + str(minNeighbors) + '.jpg',dpi=600)
   
"""
Run Main
"""   
if __name__=="__main__":
    
    imgname=['echte-freunde-teilen-mit','Webp.net-resizeimage','two girls']
    img = cv2.imread('./' + imgname[0] +'.jpg') 
    eye_detetion(img,SF,MN)

    save_image('friends_face', img, SF,MN)
    print('image saved')    
    key = cv2.waitKey(0) 
    if key == 27: #Esc key
        cv2.destroyAllWindows()
 