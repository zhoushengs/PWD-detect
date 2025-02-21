import warnings
warnings.filterwarnings('ignore')
import time
from time import strftime
from datetime import datetime
from time import gmtime
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
#C:\Users\zhang\AppData\Roaming\MobaXterm\slash\RemoteFiles\264206_10_164\yolov8-detr-faster-newneck-2stream.yaml
if __name__ == '__main__':
    # model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    model = YOLO("ultralytics/cfg/models/v8/yolov8-fasternet.yaml")
    # model = YOLO("ultralytics/cfg/models/v10/yolov10n.yaml")
    # model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')


    #model.load('yolov8n.pt') # loading pretrain weights
    dt1 = datetime.now()
    
    print(dt1.strftime('%Y-%m-%d %H:%M:%S'))
    model.train(#data="/home/zimozhou/RTDETR-main/dataset/1.yaml",# 
                data = "/home/zimozhou/RTDETR-main-3/dataset/data.yaml",
                lr0=0.001,
                lrf=0.005,
                cos_lr=True,
                seed=20,
                cache=True,
                imgsz=640,
                epochs=300,
                patience=0,
                warmup_epochs=10,
                warmup_bias_lr=0.0001,
                batch=96,
                #mosaic = 0.1,
                #close_mosaic=0,
                workers=12,
                device='0',
                #device=[0,1],
                optimizer='AdamW', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                #cls=1,
                project='runs/train',
                #name='expv8_PGI_AdamW2',
                name='exp-faster_dw',
                
                )
    dt2 = datetime.now()
    
    print(dt2.strftime('%Y-%m-%d %H:%M:%S'))
    seconds = (dt2 - dt1).seconds
    print('Traning time:', strftime('%H:%M:%S',gmtime(seconds)))