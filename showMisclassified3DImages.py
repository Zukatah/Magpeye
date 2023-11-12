import cv2 as cv
import numpy as np
import os
import csv
import pyperclip


with open('NotesAndTools/MislabeledFiles.csv', 'r') as csvfile:
    notesReader = csv.reader(csvfile, delimiter=',')
    
    for (i, row) in enumerate(notesReader):
        if len(row) == 3:
            print(i, row[0][2:-1])
            picturepath = row[0][2:-1] #"Pictures3D\\val\\2\\IMG_0249_mv_24257_3.npy"
            img = np.load(picturepath)
            newimg = cv.hconcat([img[0,:,:], img[1,:,:], img[2,:,:], img[3,:,:], img[4,:,:]])
            pyperclip.copy(str(row[0]))
            cv.imshow("File: " + str(row[0]) + " True: " + str(row[1]) + " Pred: " + str(row[2]), newimg)
            k = cv.waitKey(0)



'''
folderpath = "Pictures3D_NewlyCreated/train/3/"

for file in os.scandir(folderpath):
    if file.name.endswith(".npy") and file.name.__contains__("_0.") and not file.name.__contains__("_mv_") and file.name.__contains__("0674"):
        img = np.load(file.path)
        newimg = cv.hconcat([img[0,:,:], img[1,:,:], img[2,:,:], img[3,:,:], img[4,:,:]])
        cv.imshow("File: " + str(file.name), newimg)
        k = cv.waitKey(0)
'''