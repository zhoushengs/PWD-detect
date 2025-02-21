import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

# Load a model
 # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':
    #model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    model = YOLO("ultralytics/cfg/models/v8/yolov8-fasternet.yaml")
    #model = YOLO("ultralytics/cfg/models/v10/yolov10n.yaml")
    #model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    #model = YOLO("E:\\projects\\pytorch\\yolov11\\ultralytics\\runs\detect\\train23\\weights\\best.pt")
    #model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')

    model.train(data = 'cfg/datasets/coco8.yaml',
                #data="cfg/datasets/tree-small.yaml",
                lr0=0.002,
                lrf=0.005,
                cos_lr=True,
                seed=20,
                cache=False,
                imgsz=640,
                epochs=2,
                patience=0,
                warmup_epochs=5,
                warmup_bias_lr=0.0001,
                batch=8,
                # mosaic = 0.1,
                # close_mosaic=0,
                workers=1,
                device='0',
                #device=[0, 1],
                optimizer='AdamW',  # using SGD
                #resume="E:\\projects\\pytorch\\yolov11\\ultralytics\\runs\\detect\\train22\\weights\\last.pt", # last.pt path
                amp=False, # close amp
     )

