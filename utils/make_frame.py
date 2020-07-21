import cv2
from pathlib import Path
import os

save_dir = Path('../dataset/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images/')
img_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture("path_to_video")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

train = length * 0.8
test = length - train

i = 0
while (cap.isOpened()):
    flag, frame = cap.read()
    i += 1
    if flag == False:
        cap.release()
        break
    if i < train:
    	train_dir = img_dir.joinpath('train')
    	train_dir.mkdir(exist_ok=True)
    	cv2.imwrite(str(train_dir.joinpath('{:05}.png'.format(i))), frame)
    else:
    	test_dir = img_dir.joinpath('test')
    	test_dir.mkdir(exist_ok=True)
    	cv2.imwrite(str(test_dir.joinpath('test/{:05}.png'.format(i))), frame)
    if i%1000 == 0:
        print('Has generated %d picetures'%i)

