from skimage import data
import cv2
import matplotlib.pyplot as plt

from skimage import data
from _path import get_cut_dir

# '''Read the image data'''
dir_home = get_cut_dir('openCV_TAcademy')

# 컬러 영상 출력
# imgBGR = cv2.imread(dir_home + '/src/cat.bmp')
imgRGB = data.chelsea()        # cat 'chelsea' image, type: nd.array
imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

# 그레이스케일 영상 출력
# imgGray = cv2.imread(dir_home + '/src/cat.bmp', cv2.IMREAD_GRAYSCALE)
imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

plt.imshow(imgRGB)
plt.show()

# plt.imshow(imgGray)  # blue-green print, give it a cmap='gray
plt.imshow(imgGray, cmap='gray')
plt.show()

# 두 개의 영상을 함께 출력
# Multiple box plots on one Axes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
axes[0].imshow(imgRGB)
axes[1].imshow(imgGray, cmap='gray')

plt.show()
