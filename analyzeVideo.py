import cv2 as cv
import os

videofolder = 'Videos_Raw/'

def analyze_video(video_path):
    capture = cv.VideoCapture(video_path)
    length, width, height, fps = int(capture.get(cv.CAP_PROP_FRAME_COUNT)), int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)), capture.get(cv.CAP_PROP_FPS)
    print(f"Video: {video_path}, length: {length}, width: {width}, height: {height}, fps: {fps}")
    capture.release()

# Loop through all mp4 and mov files in videofolder and call analyze_video for each video file to display information about it
for filename in os.listdir(videofolder):
    if filename.lower().endswith(".mp4") or filename.lower().endswith(".mov"):
        video_path = os.path.join(videofolder, filename)
        analyze_video(video_path)