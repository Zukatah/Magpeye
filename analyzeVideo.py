import cv2 as cv

videoname = "IMG_0674"

capture = cv.VideoCapture('Videos_Raw/' + videoname + '.mov')
length, width, height, fps = int(capture.get(cv.CAP_PROP_FRAME_COUNT)), int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)), capture.get(cv.CAP_PROP_FPS)
print( "length", length, "width", width, "height", height, "fps", fps )

capture1 = cv.VideoCapture('Videos_Raw/' + videoname + '_Frames.mp4')
length, width, height, fps = int(capture1.get(cv.CAP_PROP_FRAME_COUNT)), int(capture1.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture1.get(cv.CAP_PROP_FRAME_HEIGHT)), capture1.get(cv.CAP_PROP_FPS)
print( "length", length, "width", width, "height", height, "fps", fps )

capture2 = cv.VideoCapture('Videos_Raw/' + videoname + '_Frames_RD.mp4')
length, width, height, fps = int(capture2.get(cv.CAP_PROP_FRAME_COUNT)), int(capture2.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture2.get(cv.CAP_PROP_FRAME_HEIGHT)), capture2.get(cv.CAP_PROP_FPS)
print( "length", length, "width", width, "height", height, "fps", fps )