import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import glob
import dlib
import urllib
import urllib.request as ur
import csv



import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import json
options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.5
}

tfnet = TFNet(options)


#******************************Boto3*********************************
from boto3.session import Session

ACCESS_KEY='AKIAIZ73P6CRR67YROJQ'
SECRET_KEY='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'
session = Session(aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
your_bucket = s3.Bucket('picpulse-imgs')
def file_exists(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


#*********************Url to Download Image**************************

def url_to_image(url):
    resp = ur.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
    return image


class FaceCV(object):
    CASE_PATH = "haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


# f=None

# res=str(type(f))

# print(res)

# if res=="<class 'NoneType'>":
#   print("None ha")

    def detect_face(self):
        count=0
        detector = dlib.get_frontal_face_detector()
        import csv
        resultFile=open("final1.xls",'a')
        wr = csv.writer(resultFile, dialect='excel')
        # wr.writerow(["Url","Class","Gender"])
        for s3_file in your_bucket.objects.all():
            # print(count)
            count=count+1
            if count>2002991:
                flag=True
                url=u'https://{0}.s3.amazonaws.com/{1}'.format("picpulse-imgs", s3_file.key)            
                url=url.encode("ascii", errors="ignore").decode()
                # print(url)
                print(count)
                bol=file_exists(url)
                if bol==True:
                    # url="https://picpulse-imgs.s3.amazonaws.com/102808803969032054.jpg"
                    frame=url_to_image(url)
                    res=""
                    res=str(type(frame))

                    # print(frame)
                    if res=="<class 'NoneType'>":
                        wr.writerow([str(url),str("None"),str("None"),"None"])
                    else:
                        faces = detector(frame, 1)
                        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_h, img_w, _ = np.shape(input_img)

                        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                        for i, d in enumerate(faces):
                            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                            xw1 = max(int(x1 - 0.4 * w), 0)
                            yw1 = max(int(y1 - 0.4 * h), 0)
                            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                            face_imgs[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.face_size, self.face_size))

                        if len(face_imgs) > 0:
                            # predict ages and genders of the detected faces
                            results = self.model.predict(face_imgs)
                            predicted_genders = results[0]
                            # print(count)
                            ages = np.arange(0, 101).reshape(101, 1)
                            predicted_ages = results[1].dot(ages).flatten()
                        else:
                            result = tfnet.return_predict(frame)
                            # print(result)
                            if len(result)>0:
                                # tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
                                # br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])

                                # data = json.loads(result)

                                clist=[]
                                for x in result:
                                    res=x['label']
                                    clist.append(str(res))
                                # print(clist)

                                # add the box and label and display it
                                if "person" in clist:
                                    label="person"
                                    # img = cv2.rectangle(frame, tl, br, (0, 255, 0), 7)
                                    # img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (5, 5, 15), 2)
                                    # cv2.imwrite("results1/out1/frame%d.jpg" % count, img)
                                    wr.writerow([str(url),str("no faces"),str("no gender"),"person"])
                                else:
                                    wr.writerow([str(url),str("no faces"),str("no gender"),"None"])

                        for i, face in enumerate(faces):
                            nn=predicted_ages[i]
                            if nn>0 and nn<=12:
                                predicted_g="0-12"
                            elif nn>12 and nn<=17:
                                predicted_g="13-17"
                            elif nn>17 and nn<=32:
                                predicted_g="18-32"
                            elif nn>32 and nn<=54:
                                predicted_g="33-54"
                            elif nn>33 and nn<=54:
                                predicted_g="55+"
                            label = "{},{}".format(predicted_g,
                                                    "Female" if predicted_genders[i][0] > 0.5 else "Male")
                            re="Female" if predicted_genders[i][0] > 0.5 else "Male"
                            # print(re)
                            # self.draw_label(frame, (face.left(), face.top()), label)

                            if flag==True:
                                result = tfnet.return_predict(frame)
                                flag=False
                            # print(result)
                            if len(result)>0:
                                # tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
                                # br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])

                                # data = json.loads(result)

                                clist=[]
                                for x in result:
                                    res=x['label']
                                    clist.append(str(res))
                                # print(clist)

                                # add the box and label and display it
                                if "person" in clist:
                                    label="person"
                                    # img = cv2.rectangle(frame, tl, br, (0, 255, 0), 7)
                                    # img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (5, 5, 15), 2)
                                    # cv2.imwrite("results1/out1/frame%d.jpg" % count, img)
                                    wr.writerow([str(url),str(predicted_g),str(re),"person"])
                                else:
                                    wr.writerow([str(url),str(predicted_g),str(re),"None"])
                            else:
                                wr.writerow([str(url),str(predicted_g),str(re),"None"])
                            # cv2.imshow('Keras Faces', frame)
                            # count = count+1
                            # wr.writerow([str(url),str(predicted_g),str(re),"person"])
                            # cv2.imwrite("results/frame%d.jpg" % count, frame)
                else:
                    print("invalid url")
            
def main():
    face = FaceCV(depth=16, width=8)

    face.detect_face()

if __name__ == "__main__":
    main()