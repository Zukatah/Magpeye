import os
import random
import numpy as np
from globalConstants import TRAINING_EXAMPLE_C0_GENERATION_LIST, TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST


'''
trainExClass = 3
trainExClassMv = 1 if trainExClass == 3 else 3
folderpath_source = "Pictures3D_Input/" + str(trainExClass) + "/"
folderpath_source_mv = "Pictures3D_Input/" + str(trainExClassMv) + "/"
folderpath_dest_train = "Pictures3D/train/" + str(trainExClass) + "/"
folderpath_dest_val = "Pictures3D/val/" + str(trainExClass) + "/"
folderpath_dest_mv_train = "Pictures3D/train/" + str(trainExClassMv) + "/"
folderpath_dest_mv_val = "Pictures3D/val/" + str(trainExClassMv) + "/"

trainingExamples = os.listdir(folderpath_source)

for i, trainingExample in enumerate(trainingExamples):
    basisname, fileExtension = os.path.splitext(trainingExample)
    teile = basisname.split('_')

    if len(teile) == 4 and int(teile[3]) == TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[0] and teile[0] != "Magpeye-Android":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
        for i in TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST: # depending on number of training examples per collision (10 for depth 30 of 3Dsamples)
            oneOfTenCopies = neuer_basisname + "_" + str(i) + fileExtension # needs to be adjusted accordingly (i*3+1 for depth 30 of 3Dsamples)
            oneOfTenCopiesMv = neuer_basisname_mv + "_" + str(i) + fileExtension # needs to be adjusted accordingly (i*3+1 for depth 30 of 3Dsamples)
            #print(i, os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            #print(i, os.path.join(folderpath_source_mv, oneOfTenCopiesMv), os.path.join((folderpath_dest_mv_train if train else folderpath_dest_mv_val), oneOfTenCopiesMv))
            os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            os.replace(os.path.join(folderpath_source_mv, oneOfTenCopiesMv), os.path.join((folderpath_dest_mv_train if train else folderpath_dest_mv_val), oneOfTenCopiesMv))
    elif len(teile) == 3 and teile[0] == "Magpeye-Android":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + str(trainExClassMv)
        trainEx = neuer_basisname + fileExtension
        trainExMv = neuer_basisname_mv + fileExtension
        os.replace(os.path.join(folderpath_source, trainEx), os.path.join((folderpath_dest_train if train else folderpath_dest_val), trainEx))
        os.replace(os.path.join(folderpath_source_mv, trainExMv), os.path.join((folderpath_dest_mv_train if train else folderpath_dest_mv_val), trainExMv))
    
    if i % 1000 == 0:
        print(i)

    #if i > 8:
    #    break
'''


'''
folderpath_source = "Pictures3D_Input/2/"
folderpath_dest_train = "Pictures3D/train/2/"
folderpath_dest_val = "Pictures3D/val/2/"

trainingExamples = os.listdir(folderpath_source)

for i, trainingExample in enumerate(trainingExamples):
    basisname, fileExtension = os.path.splitext(trainingExample)
    teile = basisname.split('_')

    if len(teile) == 4 and int(teile[3]) == TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[0] and teile[0] != "Magpeye-Android":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
        for i in TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST: # depending on number of training examples per collision (10 for depth 30 of 3Dsamples)
            oneOfTenCopies = neuer_basisname + "_" + str(i) + fileExtension # needs to be adjusted accordingly (i*3+1 for depth 30 of 3Dsamples)
            oneOfTenCopiesMv = neuer_basisname_mv + "_" + str(i) + fileExtension # needs to be adjusted accordingly (i*3+1 for depth 30 of 3Dsamples)
            os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            os.replace(os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
    elif len(teile) == 3 and teile[0] == "Magpeye-Android":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
        trainEx = neuer_basisname + fileExtension
        trainExMv = neuer_basisname_mv + fileExtension
        os.replace(os.path.join(folderpath_source, trainEx), os.path.join((folderpath_dest_train if train else folderpath_dest_val), trainEx))
        os.replace(os.path.join(folderpath_source, trainExMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), trainExMv))

    if i % 1000 == 0:
        print(i)
'''



folderpath_source = "Pictures3D_Input/0/"
folderpath_dest_train = "Pictures3D/train/0/"
folderpath_dest_val = "Pictures3D/val/0/"

trainingExamples = os.listdir(folderpath_source)

for i, trainingExample in enumerate(trainingExamples):
    basisname, fileExtension = os.path.splitext(trainingExample)
    teile = basisname.split('_')

    if len(teile) == 4 and teile[0] != "Magpeye-Android":
        if int(teile[3]) >= 0:
            train = random.random() < 0.9
            neuer_basisname = '_'.join(teile[0:4])
            neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2] + "_" + teile[3]
            oneOfTenCopies = neuer_basisname + fileExtension
            oneOfTenCopiesMv = neuer_basisname_mv + fileExtension
            #print(i, os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            #print(i, os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
            os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
            os.replace(os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
        elif int(teile[3]) == TRAINING_EXAMPLE_C0_GENERATION_LIST[0]:
            train = random.random() < 0.9
            neuer_basisname = '_'.join(teile[0:3])
            neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
            for i in TRAINING_EXAMPLE_C0_GENERATION_LIST: # depending on number of training examples per collision (10 for depth 30 of 3Dsamples)
                oneOfTenCopies = neuer_basisname + "_" + str(i) + fileExtension # needs to be adjusted accordingly (i*3+1 for depth 30 of 3Dsamples)
                oneOfTenCopiesMv = neuer_basisname_mv + "_" + str(i) + fileExtension # needs to be adjusted accordingly (i*3+1 for depth 30 of 3Dsamples)
                os.replace(os.path.join(folderpath_source, oneOfTenCopies), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopies))
                os.replace(os.path.join(folderpath_source, oneOfTenCopiesMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), oneOfTenCopiesMv))
    elif len(teile) == 3 and teile[0] == "Magpeye-Android":
        train = random.random() < 0.9
        neuer_basisname = '_'.join(teile[0:3])
        neuer_basisname_mv = '_'.join(teile[0:2]) + "_mv_" + teile[2]
        trainEx = neuer_basisname + fileExtension
        trainExMv = neuer_basisname_mv + fileExtension
        os.replace(os.path.join(folderpath_source, trainEx), os.path.join((folderpath_dest_train if train else folderpath_dest_val), trainEx))
        os.replace(os.path.join(folderpath_source, trainExMv), os.path.join((folderpath_dest_train if train else folderpath_dest_val), trainExMv))
        

    if i % 1000 == 0:
        print(i)
