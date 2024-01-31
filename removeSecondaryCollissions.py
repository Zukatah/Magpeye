import csv
import os
import sys

videos_path = "Videos_Preprocessed/"

for file_index, file in enumerate(os.scandir(videos_path)):
    if file.name.endswith(".csv"):
        suffixIndex = file.name.find("_ND")
        if suffixIndex == -1:
            continue
        videoname = file.name[0:-4]
        #print("videoname", videoname)

        labels = [[],[],[]]
        labels_ND = [[],[],[]]

        with open(videos_path + file.name) as csvfile:
            notesReader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for (i, row) in enumerate(notesReader):
                for collision in row:
                    labels[i].append(int(collision))
                labels[i].reverse()

        lastCollisionIndex = -sys.maxsize
        elementsInLabels = len(labels[0]) + len(labels[1]) + len(labels[2])
        while elementsInLabels > 0:
            l0 = sys.maxsize if len(labels[0]) == 0 else labels[0][-1]
            l1 = sys.maxsize if len(labels[1]) == 0 else labels[1][-1]
            l2 = sys.maxsize if len(labels[2]) == 0 else labels[2][-1]

            if l0 < l1 and l0 < l2:
                labels[0].pop()
                if l0 > lastCollisionIndex + 120: # should definitely not be higher! in recording sessions there are sometimes only around 2s between collisions
                    labels_ND[0].append(l0)
                    lastCollisionIndex = l0
                else:
                    print(file.name, " ", lastCollisionIndex, " ", (l0 - lastCollisionIndex))
            elif l1 < l0 and l1 < l2:
                labels[1].pop()
                if l1 > lastCollisionIndex + 120:
                    labels_ND[1].append(l1)
                    lastCollisionIndex = l1
                else:
                    print(file.name, " ", lastCollisionIndex, " ", (l1 - lastCollisionIndex))
            else:
                labels[2].pop()
                if l2 > lastCollisionIndex + 120:
                    labels_ND[2].append(l2)
                    lastCollisionIndex = l2
                else:
                    print(file.name, " ", lastCollisionIndex, " ", (l2 - lastCollisionIndex))
                
            elementsInLabels -= 1


        with open(videos_path + videoname + '_ND.csv', 'w', newline='') as csvfile_nd:
            writer = csv.writer(csvfile_nd)
            for row in labels_ND:
                writer.writerow(row)
        
        with open(videos_path + videoname + '_ND.csv', 'rb+') as file:
            file.seek(-2, os.SEEK_END)
            file.truncate()