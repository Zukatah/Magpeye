import csv
import cv2 as cv
import os

videos_path = "Videos_Preprocessed/"

for file_index, file in enumerate(os.scandir(videos_path)):
    if file.name.lower().endswith(".mp4") or file.name.lower().endswith(".mov"):
        videoname, _ = os.path.splitext(file.name)

        capture2 = cv.VideoCapture(videos_path + file.name)
        length = int(capture2.get(cv.CAP_PROP_FRAME_COUNT))
        print("length", length)

        labels = [0 for _ in range(length)]

        with open(videos_path + videoname + '_Notes_ND.csv') as csvfile: # _Notes_ND.csv suffix for notes without secondary collisions
            notesReader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for (i, row) in enumerate(notesReader):
                for collision in row:
                    labels[int(collision)] = i+1

        with open(videos_path + videoname + '_Labels_ND.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(labels)