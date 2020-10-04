"""
# functions : for a few initial path setting
"""
print(__doc__)

import os
import sys

from typing import List


NAME_HOME = 'openCV_TAcademy'


def get_cut_dir(name_cut:str) -> str:
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


DIR_HOME = get_cut_dir(NAME_HOME)
DIR_SRC = DIR_HOME + 'src/'

sys.path.insert(0, DIR_HOME)






if __name__ == '__main__':
    """# for TEST - img = None / True / False"""
    img = False

    if stop_if_none(img,message='image loading Failed!'):
        print('image is successfully loaded : COMPLETE!')
