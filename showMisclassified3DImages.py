import cv2 as cv
import numpy as np
import os
import csv
import pyperclip
from globalConstants import TRAINING_EXAMPLE_DEPTH

'''
with open('NotesAndTools/MislabeledFiles.csv', 'r') as csvfile:
#with open('NotesAndTools/CorrectlyLabeledFiles.csv', 'r') as csvfile:
    notesReader = csv.reader(csvfile, delimiter=',')
    
    for (i, row) in enumerate(notesReader):
        if len(row) == 7:
            print(i, row[0][2:-1])
            picturepath = row[0][2:-1] #"Pictures3D\\val\\2\\IMG_0249_mv_24257_3.npy"
            img = np.load(picturepath)
            newimg1 = cv.hconcat([img[i,:,:] for i in range(TRAINING_EXAMPLE_DEPTH//2)])
            newimg2 = cv.hconcat([img[i+TRAINING_EXAMPLE_DEPTH//2,:,:] for i in range(TRAINING_EXAMPLE_DEPTH//2)])
            newimg = cv.vconcat([newimg1, newimg2])
            pyperclip.copy(str(row[0]))
            cv.imshow("File: " + str(row[0]) + " True: " + str(row[1]) + " Pred: " + str(row[2]) + " C0: " + str(row[3]) + " C1: " + str(row[4]) + " C2: " + str(row[5]) + " C3: " + str(row[6]), newimg)
            k = cv.waitKey(0)
'''


folderpath = "Pictures3D/val/1/"

for file in os.scandir(folderpath):
    #if file.name.endswith(".npy") and file.name.__contains__("_0.") and not file.name.__contains__("_mv_") and file.name.__contains__("0674"):
    if file.name.endswith(".npy"):
        img = np.load(file.path)
        
        #ssi_index_list = []
        #for i in range(TRAINING_EXAMPLE_DEPTH-1):
        #    ssi_index_list.append(structural_similarity(img[i,:,:], img[i+1,:,:], channel_axis=None))
        #ssi_index_list_min = min(ssi_index_list)
        #ssi_index_list_avg = sum(ssi_index_list)/len(ssi_index_list)
        #print("ssi_index_list_min", ssi_index_list_min, "ssi_index_list_avg", ssi_index_list_avg)
        
        #if ssi_index_list_avg < 0.95:

        newimg1 = cv.hconcat([img[i,:,:] for i in range(5)])
        newimg2 = cv.hconcat([img[i+5,:,:] for i in range(5)])
        newimg = cv.vconcat([newimg1, newimg2])
        
        #newimg = cv.hconcat([img[0,:,:], img[1,:,:], img[2,:,:], img[3,:,:], img[4,:,:]])
        
        cv.imshow("File: " + str(file.name), newimg)
        k = cv.waitKey(0)
