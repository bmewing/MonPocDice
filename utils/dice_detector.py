import pickle
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open('./utils/tray_detection.sav', 'rb') as pickle_file:
    tray_detector = pickle.load(pickle_file)
with open('./utils/dice_detector.sav', 'rb') as pickle_file:
    dice_detector = pickle.load(pickle_file)


def load_image(fp):
    image = cv2.imread(fp)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def display_image(img, title='Cropped Image'):
    scale = 500 / img.shape[1]
    new_size = (int(img.shape[1] * scale),
                int(img.shape[0] * scale))
    resized = cv2.resize(img, new_size)
    cv2.imshow(title, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_clusters(clusters):
    plt.imshow(clusters, extent=(0, 100, 0, 100),
               interpolation='nearest', cmap=cm.gist_rainbow)
    plt.show()


def extract_contours(clusters, i):
    separation = ((clusters == i) * 255).astype('uint8')
    contours, _ = cv2.findContours(separation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def crop_to_tray(img):
    fc = img.reshape(-1, 3)
    clusters = tray_detector.predict(fc)
    clusters = clusters.reshape(img.shape[0], img.shape[1])
    contours = extract_contours(clusters, 1)
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    img_crop = img[min(box[:, 1]):max(box[:, 1]),
                   min(box[:, 0]):max(box[:, 0])]
    return img_crop


def find_kmeans(image, n_clus, viz_clusters=False):
    fc = image.reshape(-1, 3)
    kmeans = MiniBatchKMeans(n_clusters=n_clus,
                             random_state=0,
                             batch_size=10000)
    kmeans = kmeans.fit(fc)
    clusters = kmeans.predict(fc)
    clusters = clusters.reshape(image.shape[0], image.shape[1])
    return clusters


def find_dice(clusters, n_clus):
    dice = []
    for i in range(n_clus):
        contours = extract_contours(clusters, i)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 35000 > area > 13000:
                rect = cv2.minAreaRect(contour)
                rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), rect[2])
                (width, height) = (rect[1][0], rect[1][1])
                a = 1.25
                if a > height / width > 1 / a:
                    dice.append(rect)
    return dice


def detect_dice(img, viz_clusters):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n_clus = 4
    clusters = find_kmeans(img, n_clus=n_clus, viz_clusters=viz_clusters)
    hsv_clusters = find_kmeans(img_hsv, n_clus=n_clus, viz_clusters=viz_clusters)

    dice = find_dice(clusters, n_clus) + find_dice(hsv_clusters, n_clus)
    dice = collapse_dice(dice)

    for d in dice:
        box = cv2.boxPoints(d)
        cv2.drawContours(img, [box.reshape(-1, 1, 2).astype(np.int32)], 0, (0, 255, 0), 4)

    return dice, img


def collapse_dice(dice):
    overlap_threshold = 0.1
    clustered = []
    output = []
    poly_dice = [rect_polygon(d) for d in dice]
    counter = [i for i in range(len(dice))]
    while len(counter) > 0:
        ix = counter[0]
        batch = [ix]
        for iy in counter[1:]:
            a = poly_dice[ix]
            b = poly_dice[iy]
            if a.intersection(b).area > (overlap_threshold * max([a.area, b.area])):
                batch.append(iy)
        counter = [c for c in counter if c not in batch]
        clustered.append(batch)
    for clus in clustered:
        areas = [poly_dice[i].area for i in clus]
        which = [i for i, v in enumerate(areas) if v == min(areas)]
        output.append(dice[clus[which[0]]])

    return output


def rect_polygon(die):
    """Return a shapely Polygon describing the rectangle with centre at
    (x, y) and the given width and height, rotated by angle quarter-turns.

    """
    x = die[0][0]
    y = die[0][1]
    width = die[1][0]
    height = die[1][1]
    angle = die[2]
    w = width / 2
    h = height / 2
    p = Polygon([(-w, -h), (w, -h), (w, h), (-w, h)])
    return translate(rotate(p, angle * 90), x, y)
