import numpy as np
import cv2
import pickle
import re
import os
import argparse

from glob import glob
from tqdm import tqdm

print('Initiating dataset builder...')

# argument parsers
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--traindir', required=True, help='Input path to the training images directory')
arg_parser.add_argument('--testdir', required=True, help='Input path to the test images directory')
arg_parser.add_argument('--proto', required=True, help='Path to deploy.prototxt file')
arg_parser.add_argument('--caffemodel', required=True, help='Path to your weights.caffemodel file')
arg_parser.add_argument('--train_output', required=True, help='Training images output/save path')
arg_parser.add_argument('--test_output', required=True, help='Testing images output/save')
args = vars(arg_parser.parse_args())

# create glob object for both train and test directories
print('[INFO] Training images directory found:', args['traindir'])
print('[INFO] Testing images directory found:', args['testdir'])
train_img_glob = glob(args['traindir'] + '/*/*.jpg')
test_img_glob = glob(args['testdir'] + '/*/*.jpg')

# train and test directories
base = os.path.dirname('')
TRAIN_DIR = os.path.join(base + args['traindir'] )
TEST_DIR = os.path.join(base + args['testdir'])

# initiate model and weights for extracting faces
print('[INFO] Proto file found:', args['proto'])
print('[INFO] Caffemodel file found:', args['caffemodel'])
prototxt_file = os.path.join(base + args['proto'])
caffemodel_file = os.path.join(base + args['caffemodel'])

total = len(train_img_glob) + len(test_img_glob)
percent_train = int(len(train_img_glob)/total * 100)
percent_test = int(len(test_img_glob)/total * 100)

print("\n[INFO] Total images:", total)
print("[INFO] Total number of training images before preprocessing:", len(train_img_glob), "(", percent_train, "%)")
print("[INFO] Total number of training images before preprocessing:", len(test_img_glob), "(", percent_test, "%)")

# use cv2 built-in method cv2.dnn.readNetFromCaffe to extract faces
print('\n[STATUS] Loading face detection model...')
face_model = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)

print('\n[STATUS] Extracting training images')
CONFIDENCE = 0.5
IMG_SIZE = 224
ONE_HOT_LABEL = {'dahyun':0, 'juri':1, 'sohee':2, 'suyun':3, 'yeonhee':4, 'yunkyoung':5}
train_extracted_faces = []
test_extracted_faces = []
names = []
train_skipped = 0

# training image
for img in tqdm(train_img_glob):
    # read in the image
    a = cv2.imread(img)
    (h,w) = a.shape[:2]
    # create a blob object
    blob = cv2.dnn.blobFromImage(a, scalefactor=1.0, size=(IMG_SIZE, IMG_SIZE), 
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_model.setInput(blob)
    # detect face
    detector = face_model.forward() 
    
    # if any face is detected
    if len(detector) > 0:
        # get the index of the face
        i = np.argmax(detector[0,0,:,2])
        confidence = detector[0,0,i,2]
        
        # if the confidence is higher than the threshold we set earlier
        if confidence > CONFIDENCE:
            # extract face from a rectangle
            rect = detector[0,0,i,3:7] * np.array([w,h,w,h])
            start_x, start_y, end_x, end_y = rect.astype('int')
            
            face = a[start_y:end_y, start_x:end_x]
            
            # skip it if there is no face
            if face.size == 0:
                print('Skipping...')
                print('No face detected:', img)
                train_skipped += 1
                continue
            
            # resize the face
            face = cv2.resize(face, (IMG_SIZE,IMG_SIZE))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            cv2.rectangle(a, (start_x, start_y), (end_x, end_y), (255,255,255), 2)
            
            train_extracted_faces.append(face)
        
        # skip otherwise
        else:
            print('Skipping...')
            print('Confidence below threshold:', img)
            train_skipped += 1
            continue
    
    # give the label according to the correct image using regex
    label = re.findall('\\\\[a-z].*\\\\', img)[0]
    label = label[1:-1]
    
    names.append(ONE_HOT_LABEL[label])
    
# save to disk
print('\n[STATUS] Saving to local path:', args['train_output'])
train_pickle = {'extracted_faces': train_extracted_faces, 'names': names}
with open(args['train_output'], 'wb') as f:
    pickle.dump(train_pickle, f)

print('\n[STATUS] Extracting training images completed')
print('\n[STATUS] Extracting testting images')

# do the same for test image
test_extracted_faces = []
names = []
test_skipped = 0

for img in tqdm(test_img_glob):
    a = cv2.imread(img)
    (h,w) = a.shape[:2]
    blob = cv2.dnn.blobFromImage(a, scalefactor=1.0, size=(IMG_SIZE, IMG_SIZE), 
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_model.setInput(blob)
    detector = face_model.forward()
    
    if len(detector) > 0:
        i = np.argmax(detector[0,0,:,2])
        confidence = detector[0,0,i,2]
        
        if confidence > CONFIDENCE:
            rect = detector[0,0,i,3:7] * np.array([w,h,w,h])
            start_x, start_y, end_x, end_y = rect.astype('int')
            
            face = a[start_y:end_y, start_x:end_x]
            
            if face.size == 0:
                print('Skipping...')
                print('No face detected:', img)
                test_skipped += 1
                continue
                
            face = cv2.resize(face, (IMG_SIZE,IMG_SIZE))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            cv2.rectangle(a, (start_x, start_y), (end_x, end_y), (255,255,255), 2)
            
            test_extracted_faces.append(face)
        
        else:
            print('Skipping...')
            print('Confidence below threshold:', img)
            test_skipped += 1
            continue
    
    label = re.findall('\\\\[a-z].*\\\\', img)[0]
    label = label[1:-1]
    
    names.append(ONE_HOT_LABEL[label])
    
# save to disk
print('\n[STATUS] Saving to local path:', args['test_output'])
test_pickle = {'extracted_faces': test_extracted_faces, 'names': names}
with open(args['test_output'], 'wb') as f:
    pickle.dump(test_pickle, f)
    
print('[STATUS] Extracting training images completed')
    
print('Building dataset completed!')