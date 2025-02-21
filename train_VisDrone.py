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
print('hi')
if __name__ == '__main__':
    #model = YOLO('ultralytics/cfg/models/v8/yolov8-detr-C2f-DCNV4.yaml')
    #model = YOLO('ultralytics/cfg/models/v8/yolov8-fasternet-gold-2stream.yaml')
    #model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml') 
    #model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-DCNV4.yaml') 
    #model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-EMA.yaml')  
    #model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV4.yaml') 
    #model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8s-detr-C2f-DCNV4.yaml')
    model = YOLO("ultralytics/cfg/models/v8/yolov8s.yaml")
    #model = RTDETR("ultralytics/cfg/models/yolo-detr/yolov8-fasternet-detr-goldyolo.yaml")
    #model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8s-detr.yaml')
    #model = YOLO('ultralytics/cfg/models/yolo-detr/yolov8s-detr-faster-newneck-2stream_vis.yaml')
    #model = RTDETR('ultralytics/cfg/models/yolo-detr/yolov8-detr-faster-newneck-2stream.yaml')

    # "/lustre06/project/6020210/zimozhou/UCB/train/"
    #model.load('/home/zimozhou/RTDETR-main/runs/train/exp-Vis-yolov8-faster-gold17/weights/best.pt') # loading pretrain weights  /lustre06/project/6020210/zimozhou/VisDrone/ C0072073596-001-17  "/home/zimozhou/RTDETR-main/runs/train/exp-Vis-yolov8-faster-gold17/weights/best.pt"
    dt1 = datetime.now()
    print('hi')
    
    print(dt1.strftime('%Y-%m-%d %H:%M:%S'))
    model.train(data="/home/zimozhou/RTDETR-main/dataset/VisDrone.yaml",
                lr0=0.001,
                lrf=0.05,
                cos_lr=True,
                seed=3,
                cache=False,
                imgsz=640,
                epochs=200,
                patience=0,
                batch=48,
                warmup_epochs=5,
                warmup_bias_lr=0.0005,
                #mosaic = 0.1,
                #close_mosaic=0,
                workers=4,
                device='0',
                #device=[0,1],
                optimizer='AdamW', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                #cls=1,
                project='runs/train',
                #name='expv8_PGI_AdamW2',
                name='exp-Vis-yolov8-faster',
                
                )
    dt2 = datetime.now()
    
    print(dt2.strftime('%Y-%m-%d %H:%M:%S'))
    seconds = (dt2 - dt1).seconds
    print('Traning time:', strftime('%H:%M:%S',gmtime(seconds)))