"""
# Tesseract-ocr 설치하기
# ---------------------------
1. tesseract-ocr-w64-setup-v5.0.0-alpha.20191030.exe 파일 다운로드
   (https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20200328.exe)
2. 설치 시 "Additional script data" 항목에서 "Hangul Script", "Hangul vertical script" 항목 체크,
   "Additional language data" 항목에서 "Korean" 항목 체크.
4. 설치 후 시스템 환경변수 PATH에 Tesseract 설치 폴더 추가
   (e.g.) c:/Program Files/Tesseract-OCR
4. 설치 후 시스템 환경변수에 TESSDATA_PREFIX를 추가하고, 변수 값을 <Tesseract-DIR>/tessdata 로 설정
5. <Tesseract-DIR>/tessdata/script/ 폴더에 있는 Hangul.traineddata, Hangul_vert.traineddata 파일을
   <Tesseract-DIR>/tessdata/ 폴더로 복사
6. 명령 프롬프트 창에서 pip install pytesseract 명령 입력
"""

from typing import List

import sys
import cv2
import numpy as np

# import pytesseract    # ERRON SETUP! -> easyOCR
from _path import (DIR_SRC, get_cut_dir, stop_if_none)

# 영상 불러오기 : default = namecard1.jpg
filename = 'namecard1.jpg' if len(sys.argv) <= 1 else sys.argv[1]
dw, dh = (720, 400)             # 명함 왜곡보정 후 출력/저장 되는 사이즈

# 영상 로딩이 안될경우 시스템 종료!
src_RGB = cv2.imread(DIR_SRC + filename)
src_RGB = stop_if_none(src_RGB, message="image loading failed!")

# 입력 영상 전처리 = 그레이스케일 + 바이너리 이미지 만들기
src_gray = cv2.cvtColor(src_RGB, cv2.COLOR_BGR2GRAY)
_, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 출력 영상 설정
src_quards = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)

destin_quards = np.array([[0, 0], [0, dh], [dw, dh], [dw, 0]], np.float32)
destination = np.zeros((dh, dw), np.uint8)


def get_reorder_pts(pts:List[int]) -> List[int]:
    """ # re-dorder 4 point of rectangular"""
    # 칼럼0 -> 칼럼1 순으로 정렬한 인덱스를 반환
    idx = np.lexsort((pts[:, 1], pts[:, 0]))

    # x좌표로 정렬
    pts = pts[idx]

    if pts[0, 1] > pts[1, 1]:
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]:
        pts[[2, 3]] = pts[[3, 2]]

    return pts

# 바이너리 이미지(src_bin) 에서 외곽선 검출하고 그 중에 명함추출 하는 방법
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 외곽선(contour) 중에서 타겟 외곽선(approx)를 추출 하는 루틴!
for contour in contours:
    # 너무 작은 객체는 제외 = 면적이 1,000픽셀보다 작은 객체는 제외
    if cv2.contourArea(contour) < 10000:
        continue

    # 외곽선의 근사치 주적 알고리즘 = DP (Douglas-Peucker) algorithm
    #  - Point 수를 줄이는 방식은 Douglas-Peucker algorithm.
    #  - 임의의 폭 안쪽으로 들어오지 않는 포인트를 삭제해 나가면서 외곽선 추출
    approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.02, True)

    # 컨벡스가 아니거나 4각형(4 contour)이 아니면 제외 시킴
    if not cv2.isContourConvex(approx) or len(approx) != 4:
        continue

# 원본 이미지(src_RGB) 위에 추출된 좌표(approx)를 그린다
cv2.polylines(src_RGB, [approx], True, (0, 255, 0), 2, cv2.LINE_AA)
src_quards = get_reorder_pts(approx.reshape(4, 2).astype(np.float32))

# 퍼스펙트 왜곡을 보정한다.
pers = cv2.getPerspectiveTransform(src_quards, destin_quards)
destination = cv2.warpPerspective(src_RGB, pers, (dw, dh), flags=cv2.INTER_CUBIC)

# 보여주기 위해서, BRG 포맷을 RGB 포맷으로 변형 시칸다
# RGB퍼스팩을 추출해서 데스틴에 담는다
dst_rgb = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)
src_RGB_resized = cv2.resize(src=src_RGB, dsize=(0,0), fx=0.4, fy=0.4)

# print(pytesseract.image_to_string(dst_rgb), lang='Hangul+eng')
# print(pytesseract.image_to_string(dst_rgb))

# 가공된 이미지를 보여준다.
cv2.imshow('src_RGB', src_RGB)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)
cv2.imshow('src_RGB_resized', src_RGB_resized)
cv2.imshow('namecard_extracted', destination)

cv2.imwrite(DIR_SRC + 'resultOCR\\namecard_extracted.png', destination)

cv2.waitKey()
cv2.destroyAllWindows()






# READ TEXT RESULTS by easyOCR / PIL
import easyocr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import (Image, ImageDraw)

im = Image.open(DIR_SRC + 'resultOCR\\namecard_extracted.png')
im = stop_if_none(im, message="image loading failed!")

# easyOCR command :
# CUDA not available - defaulting to CPU. Note:
# This module is much faster with a GPU.
reader = easyocr.Reader(['en', 'ko',])
bounds = reader.readtext(DIR_SRC + 'resultOCR\\namecard_extracted.png')


def draw_bounds(image, bounds, color='yellow', width=2):
    """# from PIL import (Image, ImageDraw) needed!"""
    # Draw bounding boxes

    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

im_boxed = draw_bounds(im, bounds)                # PIL needed
im_boxed.show()


# 확률(probs)순으로 내림차순 정렬을 한다.
df_bounds = pd.DataFrame(bounds, columns=['coord', 'reads', 'probs'])
df_bounds_sort = df_bounds.sort_values(by=['probs'], ascending=False, axis=0)


print('\n\n')
print(df_bounds_sort.head())


print('\n\n')
for idx, prob, read in zip(df_bounds_sort.index,df_bounds_sort.probs, df_bounds_sort.reads):
    print(f"{prob * 100:5.2f} % ... |   {read:30}")

# To SEE : ACCURACY CHART
fig, axes = plt.subplots(2,2, figsize=(12,6))
# df_bounds.probs.plot.barh(ax=axes[0,0])
df_bounds.probs.plot(ax=axes[0,0])
df_bounds.probs.hist(bins=27, ax=axes[0,1])
sns.boxplot(df_bounds.probs, ax=axes[1,1])
plt.show()



"""
for prob in probs:
    print(prob)

probs.sort(key=lambda x: x[1])          # sort by probablity
"""
