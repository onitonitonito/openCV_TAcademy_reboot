"""
# functions : for a few initial path setting
"""
print(__doc__)

import os
import sys
import cv2

from typing import List



def get_cut_dir(name_cut:str) -> str:
    """# 폴더명까지 path를 잘라 냄 : 목적폴더 생성용 """
    dir_hereent = os.path.dirname(__file__)
    dir_cut = "".join(dir_hereent.partition(name_cut)[:2]) + '/'
    return dir_cut

def stop_if_none(object:object, message:str='') -> object:
    """# 오브젝트 로딩 실패 시(None), 시스템(sys.exit) 종료 : args=message"""
    if object is None:
        if not message:
            print(f"*** ERROR: loading failed!:'{object}' -> stop system!")
        else:
            print(f"*** ERROR:{message}")
        sys.exit()
    else:
        return object

def gstreamer_pipeline(
        sensor_id=0,
        sensor_mode=3,
        flip_method=0,
        framerate=22,
        capture_width=3280,
        capture_height=2464,
        display_width=640,
        display_height=410,
    ) -> str:
    """ return default streaming commands-line 'String'
    | MAX_Frame_rate = 22fps@3296x2512 / 60fps@
    | Pixel Count: 3280 x 2464 (active pixels) 3296 x 2512 (total pixels)
    """
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def getRGB(img):
    """# cv2 COLOR CONVERT FORMAT to BGR->RGB"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def getBGR(img):
    """# cv2 COLOR CONVERT FORMAT to RGB->BGR"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def getGray(img):
    """# cv2 COLOR CONVERT FORMAT to BGR->GRAY"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

NAME_HOME = 'openCV_TAcademy_reboot'

DIR_HOME = get_cut_dir(NAME_HOME)
DIR_SRC = DIR_HOME + 'src/'

sys.path.insert(0, DIR_HOME)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def test_01():
        """FOR TEST.01 : When img == None / True / False"""
        # img = False  # OK
        img = None   # NG

        if stop_if_none(img, message='image loading Failed!'):
            print('image is successfully loaded : COMPLETE!')

    def test_02():
        """FOR TEST.02 : TO GET = gstreamer_pipeline string"""
        _ = gstreamer_pipeline()
        print(type(_))
        print(_)

    def test_03():
        """FOR TEST.03 : to convert color in cv2 : BGR <-> RGB <-> GRAY"""
        valid_exts = 'png jpg bmp'.split()
        files = [file for file in os.listdir(DIR_SRC)
                    if len(file.split('.')) > 1 and \
                        file.split('.')[-1] in valid_exts]
        print(files)

        img_cv2 = cv2.imread(DIR_SRC + 'david.jpg')
        fig, ax = plt.subplots(1,1,figsize=(6,9))
        ax.imshow(getRGB(getGray(img_cv2)))

        plt.show()


    test_03()
