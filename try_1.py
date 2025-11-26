import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C2f-Faster-EMA.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='data_cabai.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=16,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
