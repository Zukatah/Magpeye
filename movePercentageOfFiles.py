import os
import random
import numpy as np


folderpath = "Pictures3D/val/0/"
folderpath_dest = "Pictures3D_TempStorage_Class0/val/0/"

for file in os.scandir(folderpath):
    if random.random() < 0.9007:
        os.replace(file.path, folderpath_dest + file.name)
