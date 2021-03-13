import torch
import numpy as np
import time
import face_recognition
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import cv2 as cv
from random import randint


class Detector(object):
    def __init__(self):
        self.chunjie_encoding = np.loadtxt("face-encodings/chunjie_encoding",
                                           delimiter=',')
        self.yifan_encoding = np.loadtxt("face-encodings/yifan_encoding",
                                         delimiter=',')
        self.ruohai_encoding = np.loadtxt("face-encodings/ruohai_encoding",
                                          delimiter=',')
        self.name = "Unknown"
        self.known_face_encoding = [
            self.chunjie_encoding, self.yifan_encoding, self.ruohai_encoding
        ]
        self.known_face_name = ["Chunjie Shan", "Yifan Wei", "Ruohai Hu"]
        self.img_size = 640
        self.threshold = 0.6
        self.max_frame = 160
        self.init_model()

    def init_model(self):
        self.weights = 'weights/best.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(model,
                                                   'module') else model.names
        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255))
                       for _ in self.names]

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() # 半精度
        img /= 255.0 # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(
            0.002 *
            (image.shape[0] + image.shape[1]) / 2) + 1 # line/font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv.rectangle(image,
                         c1,
                         c2,
                         color,
                         thickness=tl,
                         lineType=cv.LINE_AA)
            tf = max(tl - 1, 1) # font thickness
            t_size = cv.getTextSize(cls_id, 0, fontScale=tl / 3,
                                    thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv.rectangle(image, c1, c2, color, -1, cv.LINE_AA) # filled
            cv.putText(image,
                       '{}'.format(cls_id), (c1[0], c1[1] - 2),
                       0,
                       tl / 3, [225, 255, 255],
                       thickness=tf,
                       lineType=cv.LINE_AA)
        return image

    def recognition(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(
            0.002 *
            (image.shape[0] + image.shape[1]) / 2) + 1 # line/font thickness

        tf = max(tl - 1, 1) # font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            face_location = [(y1, x2, y2, x1)]
            face_encoding = face_recognition.face_encodings(
                image, face_location)
            if face_location:
                for (y1, x2, y2,
                     x1), face_encoding in zip(face_location, face_encoding):
                    matches = face_recognition.compare_faces(
                        self.known_face_encoding, face_encoding)
                    face_distances = face_recognition.face_distance(
                        self.known_face_encoding, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        self.name = self.known_face_name[best_match_index]
                    else:
                        self.name = "Unknown"
                    t_size = cv.getTextSize(self.name,
                                            0,
                                            fontScale=tl / 3,
                                            thickness=tf)[0]
                    c1, c2 = (x1, y2), (x2, y1)
                    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
                    cv.rectangle(image, c1, c2, (0, 0, 255), -1,
                                 cv.LINE_AA) # filled
                    image = cv.putText(image,
                                       self.name, (x1, y2 + 20),
                                       0,
                                       tl / 3, (255, 255, 255),
                                       thickness=2)

        return image

    def detect(self, im):

        im0, img = self.preprocess(im)

        curr_time = time.time()
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        non_mask_boxes = []
        image_info = {}
        count = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))
                    if cls_id == 1:
                        non_mask_boxes.append((x1, y1, x2, y2, lbl, conf))
                    count += 1
                    key = '{}-{:02}'.format(lbl, count)
                    image_info[key] = [
                        '{}×{}'.format(x2 - x1, y2 - y1),
                        np.round(float(conf), 3)
                    ]

        im = self.plot_bboxes(im, pred_boxes)
        if len(non_mask_boxes) > 0:
            im = self.recognition(im, non_mask_boxes)
        infer_time = time.time() - curr_time
        print("The inference time is {:.2f} ms".format(infer_time * 1000))
        return im, image_info


if __name__ == "__main__":
    detector = Detector()
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            frame, info = detector.detect(frame)
            cv.imshow("Video", frame)
            c = cv.waitKey(1)
            if c == 27:
                break
        else:
            break
    cap.release()
    # image = cv.imread("./data/brain/images/train/030.png")
    # image, image_info = detector.detect(image)
    # cv.imshow("Frame", image)
    # cv.waitKey(0)
