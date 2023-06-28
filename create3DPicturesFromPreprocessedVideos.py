import cv2 as cv
import csv
import numpy as np
import os


videos_path = "Videos_Preprocessed/"
pictures_path = "Pictures3D_NewlyCreated/train/"
video_index = 0

for file_index, file in enumerate(os.scandir(videos_path)):
    if file.name.endswith(".mp4"):
        suffixIndex = file.name.find("_Frames_RD")
        if suffixIndex == -1:
            raise ValueError("Suffix '_Frames_RD' has to occur in each video file name. This is not true for video " + file.name)
        videoname = file.name[0:suffixIndex]


        print("Load labels for all frames of video", file_index, ":", videoname)
        labels = []
        with open(videos_path + videoname + '_Labels.csv') as csvfile:
            labelsReader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            labels = next(labelsReader)


        print("Loop through all frames of video", file_index, ":", videoname, "in order to create training samples (image series)")
        capture = cv.VideoCapture(videos_path + videoname + '_Frames_RD.mp4')
        frameIndex = 0
        frameIndex_lastCollisionIndex = -1
        waitingStepCounter = 1
        latest9Frames = []
        latest9Frames_mv = [] # mv = mirrored vertically
        while True:
            status, frame = capture.read()
            
            if not status:
                break

            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayFrameResized = cv.resize(grayFrame, (270, 480), interpolation = cv.INTER_LANCZOS4)
            grayFrameResized_mv = cv.flip(grayFrameResized, 1)

            if frameIndex >= 9:
                latest9Frames.pop(0)
                latest9Frames_mv.pop(0)

            latest9Frames.append(grayFrameResized)
            latest9Frames_mv.append(grayFrameResized_mv)

            # Create image series for non-collisions
            # This structure always makes sure that non-collision examples don't contain the frame of a collision, the previous frame or the next frame
            if labels[frameIndex] != 0:
                frameIndex_lastCollisionIndex = frameIndex
            if frameIndex - frameIndex_lastCollisionIndex >= 7:
                if waitingStepCounter % 60 == 0:
                    collisionPicture = np.stack((latest9Frames[3], latest9Frames[4], latest9Frames[5], latest9Frames[6], latest9Frames[7]), axis=0) # not np.concatenate((l9F[3],l9F[4],l9F[5],l9F[6],l9F[7]), axis=1)
                    np.save(pictures_path + "0/" + videoname + "_" + str(frameIndex-3), collisionPicture)
                    collisionPicture_mv = np.stack((latest9Frames_mv[3], latest9Frames_mv[4], latest9Frames_mv[5], latest9Frames_mv[6], latest9Frames_mv[7]), axis=0)
                    np.save(pictures_path + "0/" + videoname + "_mv_" + str(frameIndex-3), collisionPicture_mv)
                waitingStepCounter += 1

            # Create image series for collisions
            if frameIndex >= 8:
                curLabel = int(labels[frameIndex-4])
                if curLabel != 0:
                    for i in range(5):
                        collisionPicture = np.stack((latest9Frames[i], latest9Frames[i+1], latest9Frames[i+2], latest9Frames[i+3], latest9Frames[i+4]), axis=0)
                        np.save(pictures_path + str(curLabel) + "/" + videoname + "_" + str(frameIndex-4) + "_" + str(i), collisionPicture)
                        collisionPicture_mv = np.stack((latest9Frames_mv[i], latest9Frames_mv[i+1], latest9Frames_mv[i+2], latest9Frames_mv[i+3], latest9Frames_mv[i+4]), axis=0)
                        np.save(pictures_path + str(4-curLabel) + "/" + videoname + "_mv_" + str(frameIndex-4) + "_" + str(i), collisionPicture_mv)

            # Keep track of frame index (and print to see progress)
            if frameIndex % 1000 == 0:
                print("videoname", videoname, "frameIndex", frameIndex)
            frameIndex += 1

        capture.release()