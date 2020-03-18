import cv2
import glob
import numpy as np


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)
    initted, baseline = cam.read()
    recent = []
    while True:
        ret_val, img = cam.read()
        if len(recent)== 3:
            del recent[0]
        recent.append(cv2.subtract(img, baseline))
        gray = cv2.cvtColor(recent[-1], cv2.COLOR_BGR2GRAY)
        i+=1
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=False)


sim = []
pics = glob.glob('images/webcam/*')
pics.sort()
baseline = cv2.imread('images/webcam/00000.png')
for p in pics:
    img = cv2.imread(p)
    gray = cv2.cvtColor(cv2.subtract(img, baseline), cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()
    sim.append(np.mean(gray <= 90))
sim = sim[:182]
print(sim[107:109])


if __name__ == '__main__':
    sim = []
    pics = glob.glob('images/webcam/*')
    baseline = cv2.imread('images/webcam/00000.png')
    for img in pics:
        gray = cv2.cvtColor(cv2.subtract(img, baseline), cv2.COLOR_BGR2GRAY)
        gray = gray.flatten()
        sim.append(np.mean(gray <= 100))
