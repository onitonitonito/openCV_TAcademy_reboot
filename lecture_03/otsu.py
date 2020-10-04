"""
# BINARY IMAGE : AUTO THRESHOLD VALUE - OTSU-ALGORITHMS
"""
# th = threshold_value
# dst = destination_image

import sys
import cv2

from _path import get_cut_dir

print("*** PRESS 'ESC' key to Next image!", "\n\n")

dir_src = get_cut_dir('openCV_TAcademy') + '/src/'
filenames = [
        'namecard1.jpg',
        'namecard2.jpg',
        'namecard3.jpg',
    ]


for filename in filenames:
    src = cv2.imread(dir_src + filename, cv2.IMREAD_COLOR)

    # 이미지 로딩 실패! 시스템 종료
    if src is None:
        print('Image load failed!')
        sys.exit()


    src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    th, dst = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f'threshold (AUTO) =', th)

    # 사진출력
    cv2.imshow('dstination : binary-image Threshold', dst)
    cv2.imshow('src_gray', src_gray)
    cv2.imshow('src_resize', src)

    # 인도우 위치조정
    cv2.moveWindow('dstination : binary-image Threshold', x=550 , y=50 )
    cv2.moveWindow('src_gray', x=0 , y=400 )
    cv2.moveWindow('src_resize', x=0 , y=50 )

    while True:
        if cv2.waitKey() == 27:   # ESC=27
            break

cv2.destroyAllWindows()
