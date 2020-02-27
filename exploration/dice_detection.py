import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import matplotlib.cm as cm
import pickle

frame = cv2.imread('../images/pre_cropped.jpg')
img = cv2.GaussianBlur(frame, (5, 5), 0)
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

fc = img.reshape(-1, 3)

kmeans = MiniBatchKMeans(n_clusters=5,
                         random_state=0,
                         batch_size=10000)
kmeans = kmeans.fit(fc)
pickle.dump(kmeans, open('../utils/tray_detection.sav', 'wb'))
clus = kmeans.predict(fc)
clus = clus.reshape(frame.shape[0], frame.shape[1])

# print(clus[500, 1000])

# plt.imshow(clus, extent=(0, frame.shape[1], 0, frame.shape[0]),
#            interpolation='nearest', cmap=cm.gist_rainbow)
# plt.show()

separated = (clus == clus[500, 1000]) * 255
separated = separated.astype('uint8')
contours, hierarchy = cv2.findContours(separated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

img_crop = frame[min(box[:, 1]):max(box[:, 1]),
                 min(box[:, 0]):max(box[:, 0])]

# scale = 0.4
# new_size = (int(img_crop.shape[1] * scale),
#             int(img_crop.shape[0] * scale))
# resized = cv2.resize(img_crop, new_size)
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img2 = cv2.GaussianBlur(img_crop, (7, 7), 0)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

fc2 = img2.reshape(-1, 3)

n_clus = 4
kmeans2 = MiniBatchKMeans(n_clusters=n_clus,
                          random_state=0,
                          batch_size=10000)
kmeans2 = kmeans2.fit(fc2)
pickle.dump(kmeans2, open('../utils/dice_detector.sav', 'wb'))
clus2 = kmeans2.predict(fc2)
clus2 = clus2.reshape(img2.shape[0], img2.shape[1])

# plt.imshow(clus2, extent=(0, img2.shape[1], 0, img2.shape[0]),
#            interpolation='nearest', cmap=cm.gist_rainbow)
# plt.show()

dice = []
for i in range(n_clus):
    separated = (clus2 == i) * 255
    separated = separated.astype('uint8')
    # plt.imshow(separated)
    # plt.show()
    contours, hierarchy = cv2.findContours(separated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 35000 > area > 20000:
            rect = cv2.minAreaRect(contour)
            rect = ((rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), rect[2])
            (width, height) = (rect[1][0], rect[1][1])
            a = 1.25
            if a > height / width > 1/a:
                dice.append(rect)

for d in dice:
    box = cv2.boxPoints(d)
    cv2.drawContours(img_crop, [box.reshape(-1, 1, 2).astype(np.int32)], 0, (0, 255, 0), 4)

scale = 0.4
new_size = (int(img_crop.shape[1] * scale),
            int(img_crop.shape[0] * scale))
resized = cv2.resize(img_crop, new_size)
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
