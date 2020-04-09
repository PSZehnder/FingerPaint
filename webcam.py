import torch
from minimalyolo.models import Darknet
import cv2
from PIL import Image
from torchvision import transforms
from minimalyolo.utils import pad_to_square, resize, load_classes, non_max_suppression, rescale_boxes
import numpy as np
from time import sleep, time

class WebcamYolo:

    def __init__(self, config, weights, class_labels, webcam=True, fps=24,
                 denoise=False, imsize=416, confscore=0.5, nms_thresh=0.4, resolution=(640, 480)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Darknet(config, img_size=416).to(self.device)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

        self.resolution = resolution
        if webcam:
            self.initwebcam(fps, denoise, self.resolution)
        self.imsize = imsize #detection scale
        self.fps = fps

        self.classes = load_classes(class_labels)

        self.confscore = confscore
        self.nms_thresh = nms_thresh

        self.last_prediction = time()
        print('initialized!')
        sleep(1)


    def initwebcam(self, fps=24, denoise=False, resolution=(640, 480)):
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FPS, fps)
        self.webcam.set(3, resolution[0])
        self.webcam.set(4, resolution[1])
        self.denoise = denoise
        self.colors = [(128, 0, 0), (0, 128, 0)] #idk fix this sometime


    def getframe(self):
        rval, frame = self.webcam.read()
        if not rval: raise IOError('could not read webcam %s' % self.webcam)
        if self.denoise:
            frame = cv2.blur(frame, (5, 5))
        return frame

    @staticmethod
    def convertimage(image):
        if isinstance(image, Image.Image):
            return image
        try:
            image = Image.fromarray(image)
            return image
        except:
            raise TypeError('image must be np array or PIL image')

    def _im2tensor(self, im):
        img = transforms.ToTensor()(im)
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.imsize).unsqueeze(0)
        return img.to(self.device)

    def _convertdetections(self, detections):
        out = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
            out_dict = {
                'x1' : x1,
                'x2' : x2,
                'y1' : y1,
                'y2' : y2,
                'conf' : conf,
                'cls_conf': cls_conf,
                'class': self.classes[int(cls_pred)],
                'color': self.colors[int(cls_pred)]
            }
            out.append(out_dict)
        return out

    def detect(self, image):
        new_image = self.convertimage(image)
        if time() - self.last_prediction < 1 / self.fps:
            return None
        self.last_prediction = time()
        imtensor = self._im2tensor(new_image)
        with torch.no_grad():
            detections = self.model(imtensor).to('cpu')
            detections = non_max_suppression(detections, self.confscore, self.nms_thresh)[0]
            if detections is not None:
                detections = rescale_boxes(detections, self.imsize, self.resolution)
                detections = self._convertdetections(detections)
                print(detections[0])
        return detections

    def drawdetection(self, image, detections):
        if not detections:
            return image
        for det in detections:
            x1, x2, y1, y2 = det['x1'], det['x2'], det['y1'], det['y2']
            cv2.rectangle(image, (x1, y1) , (x2, y2), color=det['color'])
            cv2.putText(image, '%s: %.2f' % (det['class'], det['conf']),
                        (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (256, 256, 256), 2)
        return image

    def run(self):
        cv2.namedWindow('yolowebcam')
        while True:
            out = self.getframe()
            detection = self.detect(out)
            print(type(detection))
            out = self.drawdetection(out, detection)
            cv2.imshow('yolowebcam', np.array(out))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()









