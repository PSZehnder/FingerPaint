from webcam import WebcamYolo
import cv2

webyolo = WebcamYolo(config='./models/yolo/yolov3-custom.cfg', weights='./models/yolo/yolov3_ckpt_34.pth',
                     fps=60, class_labels='./models/yolo/classes.names', confscore=0.5)
print(webyolo.device)
webyolo.run()