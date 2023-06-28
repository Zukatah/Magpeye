import csv
import cv2 as cv
import os

videos_path = "Videos_Preprocessed/"

for file_index, file in enumerate(os.scandir(videos_path)):
    if file.name.endswith(".mp4"):
        suffixIndex = file.name.find("_Frames_RD")
        if suffixIndex == -1:
            raise ValueError("Suffix '_Frames_RD' has to occur in each video file name. This is not true for video " + file.name)
        videoname = file.name[0:suffixIndex]

        capture2 = cv.VideoCapture(videos_path + videoname + '_Frames_RD.mp4')
        length = int(capture2.get(cv.CAP_PROP_FRAME_COUNT))
        print( "length", length)

        labels = [0 for _ in range(length)]

        with open(videos_path + videoname + '_Notes.csv') as csvfile:
            notesReader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for (i, row) in enumerate(notesReader):
                for collision in row:
                    labels[int(collision)] = i+1

        with open(videos_path + videoname + '_Labels.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(labels)