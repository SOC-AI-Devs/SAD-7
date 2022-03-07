import time
from turtle import width
import numpy as np
import torch 
import cv2
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression

camera = 0
weights = 'last.pt'
width , height = (416 , 352)

device = torch.device('cpu')

model = attempt_load(weights , map_location = device)
stride = int(model.stride.max())
cudnn.benchmark = True
names = model.module.names if hasattr(model , 'module') else model.names
 
cap = cv2.VideoCapture(camera)

while(cap.isOpened()):
    time.sleep(0.2)
    ret , frame_  = cap.read()
    frame = cv2.resize(frame_, (width , height), interpolation= cv2.INTER_AREA)
    
    if ret == True:
        img = torch.from_numpy(frame).float().to(device).permute(2,0,1)
        img /= 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        pred = model(img, augment = False)[0]
        pred = non_max_suppression(pred , 0.35 , 0.35 , agnostic=True)
        
        for det in pred:
            detections = []
            for d in det:
                x1 , y1 , x2 , y2 , conf_, class_ = int(d[0]) , int(d[1]), int(d[2]), int(d[3]), d[4], int(d[5])
                frame = cv2.rectangle(frame , (x1, y1) , (x2 , y2), (0 , 255 , 0),2)
                frame = cv2.putText(frame , str(names[class_]) , (x1 , y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(0 , 255 , 0), 1)
                object_name = names[class_]
                detections.append(object_name)
                
                text = f'{detections[0]} detected'
                print(text)
            
    cv2.imshow('frame' , frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()              