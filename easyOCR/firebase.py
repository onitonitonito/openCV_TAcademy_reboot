"""
# json read/dump test : -> pd.DataFrame
"""
print(__doc__)

from typing import List

import json
import pandas as pd
import matplotlib.pyplot as plt

from _path import (DIR_HOME, get_cut_dir)

dir_here = DIR_HOME + 'easyOCR_example\\'

filename_dir = dir_here+'firebase.json'
removes = ['{', '}', '[', ']', ',','\n']

def getMultiReplaceTreat( targets:List[str], removes:List[str], substitude:str) -> None:
    """# HELPER() for getJson2Array : multiple replacement """
    _s = []
    for target in targets:
        target = target.strip()

        for remove in removes:
            target = target.replace(remove, substitude)

        _s.append(target)
    return _s

def getJson2Array(filename_dir:str, removes:List[str], echo=0) -> List[str]:
    """# get string array"""
    with open(file=filename_dir, mode='r', encoding='utf8') as file:
        data = file.readlines()

    array = getMultiReplaceTreat(targets=data, removes=removes, substitude='')

    if echo:      # for TEST
        for arr in array:
            print(dat, end="")

    return array


_s = getJson2Array(filename_dir, removes=removes)
[print(_, end="\n") for _ in _s]   # for TEST
