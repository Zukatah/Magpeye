import cv2 as cv
import csv
import numpy as np
import os
from globalConstants import TRAINING_EXAMPLE_DEPTH, TRAINING_EXAMPLE_C0_GENERATION_FRAME_INTERVAL, TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST, TRAINING_EXAMPLE_C0_GENERATION_LIST, TRAINING_EXAMPLE_HEIGHT, TRAINING_EXAMPLE_WIDTH
from skimage.metrics import structural_similarity


videos_path = "Videos_Preprocessed/"
pictures_path = "Pictures3D_NewlyCreated/train/"
video_index = 0

for file_index, file in enumerate(os.scandir(videos_path)):
    if file.name.lower().endswith(".mp4") or file.name.lower().endswith(".mov"):
        suffixIndex = file.name.find("_Frames_RD")
        if suffixIndex != -1:
            #raise ValueError("Suffix '_Frames_RD' has to occur in each video file name. This is not true for video " + file.name)
            videoname = file.name[0:suffixIndex]
        else:
            videoname, _ = os.path.splitext(file.name)


        print("Load labels for all frames of video", file_index, ":", videoname)
        labels = []
        with open(videos_path + videoname + '_Labels_ND.csv') as csvfile:
            labelsReader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            labels = next(labelsReader)


        print("Loop through all frames of video", file_index, ":", videoname, "in order to create training samples (image series)")
        capture = cv.VideoCapture(videos_path + file.name)
        frameIndex = 0
        frameIndex_lastCollisionIndex = -1
        waitingStepCounter = 1
        latestFrames = []
        latestFrames_mv = [] # mv = mirrored vertically
        latestFramesSize = TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[-1] + TRAINING_EXAMPLE_DEPTH - TRAINING_EXAMPLE_C0_GENERATION_LIST[0]
        while True:
            status, frame = capture.read()
            
            if not status:
                break

            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayFrameResized = cv.resize(grayFrame, (TRAINING_EXAMPLE_WIDTH, TRAINING_EXAMPLE_HEIGHT), interpolation = cv.INTER_LANCZOS4)
            grayFrameResized_mv = cv.flip(grayFrameResized, 1)

            if frameIndex >= latestFramesSize:
                latestFrames.pop(0)
                latestFrames_mv.pop(0)

            latestFrames.append(grayFrameResized)
            latestFrames_mv.append(grayFrameResized_mv)

            # Create image series for non-collisions
            # This structure always makes sure that non-collision examples don't contain the frame of a collision, the previous frame or the next frame
            if labels[frameIndex] != 0:
                frameIndex_lastCollisionIndex = frameIndex
            if frameIndex - frameIndex_lastCollisionIndex >= TRAINING_EXAMPLE_DEPTH + 105: # we don't want training examples containing secondary shuttle collissions, so we wait for 1.75s (used to be 2 frames since we included secondar collisions)
                if waitingStepCounter % TRAINING_EXAMPLE_C0_GENERATION_FRAME_INTERVAL == 0:
                    # TODO: INCLUDE STRUCTURAL SIMILARITY INDEX!!!
                    #collisionPicture = np.stack((latestFrames[TRAINING_EXAMPLE_DEPTH-2+j] for j in range(TRAINING_EXAMPLE_DEPTH)), axis=0) # not np.concatenate((l9F[3],l9F[4],l9F[5],l9F[6],l9F[7]), axis=1)
                    #np.save(pictures_path + "0/" + videoname + "_" + str(frameIndex-(TRAINING_EXAMPLE_DEPTH-2)), collisionPicture)
                    #collisionPicture_mv = np.stack((latestFrames_mv[TRAINING_EXAMPLE_DEPTH-2+j] for j in range(TRAINING_EXAMPLE_DEPTH)), axis=0)
                    #np.save(pictures_path + "0/" + videoname + "_mv_" + str(frameIndex-(TRAINING_EXAMPLE_DEPTH-2)), collisionPicture_mv)

                    collisionPicture = np.stack([latestFrames[latestFramesSize-TRAINING_EXAMPLE_DEPTH-1+j] for j in range(TRAINING_EXAMPLE_DEPTH)], axis=0) # not np.concatenate((l9F[3],l9F[4],l9F[5],l9F[6],l9F[7]), axis=1)
                    collisionPicture_mv = np.stack([latestFrames_mv[latestFramesSize-TRAINING_EXAMPLE_DEPTH-1+j] for j in range(TRAINING_EXAMPLE_DEPTH)], axis=0)
                    ssi_index_list = []
                    for i in range(TRAINING_EXAMPLE_DEPTH-1):
                        ssi_index_list.append(structural_similarity(collisionPicture[i,:,:], collisionPicture[i+1,:,:], channel_axis=None))
                    ssi_index_list_avg = sum(ssi_index_list)/len(ssi_index_list)
                    np.save(pictures_path + "0/" + videoname + "_" + str(frameIndex-TRAINING_EXAMPLE_DEPTH-1) + "_" + str(int(ssi_index_list_avg*10000)), collisionPicture)
                    np.save(pictures_path + "0/" + videoname + "_mv_" + str(frameIndex-TRAINING_EXAMPLE_DEPTH-1) + "_" + str(int(ssi_index_list_avg*10000)), collisionPicture_mv)
                waitingStepCounter += 1

            # Create image series for collisions
            if frameIndex >= latestFramesSize-1:
                curLabel = int(labels[frameIndex-TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[-1]])
                if curLabel != 0:
                    for i in TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST:
                        collisionPicture = np.stack([latestFrames[i-TRAINING_EXAMPLE_C0_GENERATION_LIST[0]+j] for j in range(TRAINING_EXAMPLE_DEPTH)], axis=0)
                        np.save(pictures_path + str(curLabel) + "/" + videoname + "_" + str(frameIndex-TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[-1]) + "_" + str(i), collisionPicture)
                        collisionPicture_mv = np.stack([latestFrames_mv[i-TRAINING_EXAMPLE_C0_GENERATION_LIST[0]+j] for j in range(TRAINING_EXAMPLE_DEPTH)], axis=0)
                        np.save(pictures_path + str(4-curLabel) + "/" + videoname + "_mv_" + str(frameIndex-TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[-1]) + "_" + str(i), collisionPicture_mv)
                    for i in TRAINING_EXAMPLE_C0_GENERATION_LIST:
                        collisionPicture = np.stack([latestFrames[i-TRAINING_EXAMPLE_C0_GENERATION_LIST[0]+j] for j in range(TRAINING_EXAMPLE_DEPTH)], axis=0)
                        np.save(pictures_path + str(0) + "/" + videoname + "_" + str(frameIndex-TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[-1]) + "_" + str(i), collisionPicture)
                        collisionPicture_mv = np.stack([latestFrames_mv[i-TRAINING_EXAMPLE_C0_GENERATION_LIST[0]+j] for j in range(TRAINING_EXAMPLE_DEPTH)], axis=0)
                        np.save(pictures_path + str(0) + "/" + videoname + "_mv_" + str(frameIndex-TRAINING_EXAMPLE_C1C2C3_GENERATION_LIST[-1]) + "_" + str(i), collisionPicture_mv)

            # Keep track of frame index (and print to see progress)
            if frameIndex % 3600 == 0:
                print("Videoname:", videoname, " Minute:", (frameIndex//3600))
            frameIndex += 1

        capture.release()