import os
import random
import numpy as np


'''
folderpath = "Pictures3D/val/0/"
folderpath_dest = "Pictures3D_TempStorage_Class0/val/0/"

for file in os.scandir(folderpath):
    if random.random() < 0.9007:
        os.replace(file.path, folderpath_dest + file.name)
'''


'''
folderpath_source = "Pictures3D_Input/3/"
folderpath_source_mv = "Pictures3D_Input/1/"
folderpath_dest_train = "Pictures3D/train/3/"
folderpath_dest_val = "Pictures3D/val/3/"
folderpath_dest_mv_train = "Pictures3D/train/1/"
folderpath_dest_mv_val = "Pictures3D/val/1/"

dateien = os.listdir(folderpath_source)

for i, datei in enumerate(dateien):
    basisname, erweiterung = os.path.splitext(datei)
    teile = basisname.split('_')

    if len(teile) == 4 and teile[3] == "1":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
        for i in range(10):
            oneOfTenCopies = neuer_basisname + "_" + str(i*3+1) + erweiterung
            oneOfTenCopiesMv = neuer_basisname_mv + "_" + str(i*3+1) + erweiterung
            #print(i, os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            #print(i, os.path.join(folderpath_source_mv, oneOfTenCopiesMv), os.path.join((folderpath_dest_mv_train if train else folderpath_dest_mv_val), oneOfTenCopiesMv))
            os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            os.replace(os.path.join(folderpath_source_mv, oneOfTenCopiesMv), os.path.join((folderpath_dest_mv_train if train else folderpath_dest_mv_val), oneOfTenCopiesMv))
    
    if i % 1000 == 0:
        print(i)

    #if i > 8:
    #    break
'''


'''
folderpath_source = "Pictures3D_Input/2/"
folderpath_dest_train = "Pictures3D/train/2/"
folderpath_dest_val = "Pictures3D/val/2/"

dateien = os.listdir(folderpath_source)

for i, datei in enumerate(dateien):
    basisname, erweiterung = os.path.splitext(datei)
    teile = basisname.split('_')

    if len(teile) == 4 and teile[3] == "1":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
        for i in range(10):
            oneOfTenCopies = neuer_basisname + "_" + str(i*3+1) + erweiterung
            oneOfTenCopiesMv = neuer_basisname_mv + "_" + str(i*3+1) + erweiterung
            #print(i, os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            #print(i, os.path.join(folderpath_source_mv, oneOfTenCopiesMv), os.path.join((folderpath_dest_mv_train if train else folderpath_dest_mv_val), oneOfTenCopiesMv))
            os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            os.replace(os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
    
    if i % 1000 == 0:
        print(i)
'''


folderpath_source = "Pictures3D_TempStorage_Class0/"
folderpath_dest_train = "Pictures3D/train/0/"
folderpath_dest_val = "Pictures3D/val/0/"

dateien = os.listdir(folderpath_source)

for i, datei in enumerate(dateien):
    basisname, erweiterung = os.path.splitext(datei)
    teile = basisname.split('_')

    if len(teile) == 4:
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:4])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2] + "_" + teile[3]
        oneOfTenCopies = neuer_basisname + erweiterung
        oneOfTenCopiesMv = neuer_basisname_mv + erweiterung
        #print(i, os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
        #print(i, os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
        os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
        os.replace(os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
    
    if i % 1000 == 0:
        print(i)