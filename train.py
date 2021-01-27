import numpy as np
import cv2
import pickle
import re
import os
import argparse
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50
from torchvision import datasets, transforms

from model import Model

print('Initiating model training...')

# argument parsers
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--traindir', required=True, help='Input path to the training images directory')
arg_parser.add_argument('--testdir', required=True, help='Input path to the test images directory')
arg_parser.add_argument('--lr', required=True, help='Model learning rate')
arg_parser.add_argument('--epochs', required=True, help='Number of epochs')
arg_parser.add_argument('--batch_size', required=True, help='Batch size value')
arg_parser.add_argument('--save_path', required=False, help='Training weighst output/save path')
args = vars(arg_parser.parse_args())

print('[INFO] Dataset found: {} (training) and {} (testing)'.format(args['traindir'], args['testdir']))

# load the saved extracted faces pickle
with open(args['traindir'], 'rb') as f:
    train_face_loader = pickle.load(f)

with open(args['testdir'], 'rb') as f:
    test_face_loader = pickle.load(f)
    
raw_train_data = list(train_face_loader.values())
raw_test_data = list(test_face_loader.values())

train_features = raw_train_data[0]
train_labels = raw_train_data[1]
test_features = raw_test_data[0]
test_labels = raw_test_data[1]

IMG_SIZE = 224
m = len(train_features)
m_test = len(test_features)
c = 3

# transform train and test data into numpy arrays
x_train = np.asarray(train_features)
x_test = np.asarray(test_features)
# don't forget to reshape features to "color channel first" shape
x_train_batch_first = x_train.reshape([m,c,IMG_SIZE,IMG_SIZE])
x_test_batch_first = x_test.reshape([m_test,c,IMG_SIZE,IMG_SIZE])
y_train = np.asarray(train_labels)
y_test = np.asarray(test_labels)

BATCH_SIZE = int(args['batch_size'])
NUM_WORKERS = 0

# train_transform = transforms.Compose([transforms.RandomVerticalFlip(0.3),
#                                 transforms.RandomRotation(30),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                 transforms.ToTensor()])

# finally, we need to transform the data into Tensor and they are ready to be trained
train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train_batch_first), torch.from_numpy(y_train))
test_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test_batch_first), torch.from_numpy(y_test))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

# check for CUDA and instantiate the model
use_cuda = torch.cuda.is_available()
model = Model()

# move the model to GPU if available
if use_cuda:
    print('[INFO] Training using GPU:', torch.cuda.get_device_name(0))
    model = model.cuda()
else:
    print('[INFO] No GPU devices found, training using CPU')

# specify criterion and optimizer
# feel free to experiment with various hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=float(args['lr']), weight_decay=1e-5)

print('[INFO] Training hyperparameters: {} epochs | {} learning rate | {} batch size'.format(args['epochs'], args['lr'], args['batch_size']))

for i in range(int(args['epochs'])):
    training_loss = 0.0
        
    model.train()
    for batch, (data, lbl) in enumerate(train_loader):
        data = data.to(torch.float32)
        lbl = lbl.to(torch.long)
        if use_cuda:
            data, lbl = data.cuda(), lbl.cuda()
                
        optimizer.zero_grad()
            
        output = model(data)
        loss = criterion(output, lbl)
        loss.backward()
        optimizer.step()
            
        training_loss = training_loss + ((1/(batch+1)) * (loss.data - training_loss))
        
    print("Epoch:", i+1, "| Training Loss:", training_loss)

test_loss = 0.0
correct = 0
total_data = 0
    
model.eval()
for batch_idx, (feature, lbl) in enumerate(test_loader):
    feature = feature.to(torch.float32)
    lbl = lbl.to(torch.long)
    if use_cuda:
        feature, lbl = feature.cuda(), lbl.cuda()
            
    output = model(feature)
    loss = criterion(output, lbl)
    test_loss = test_loss + (1/(batch_idx+1)) * (loss.data - test_loss)
    total_data = total_data + feature.size(0)
        
    pred = output.data.max(1)[1]
    correct += np.sum(np.squeeze(pred.eq(lbl.data.view_as(pred))).cpu().numpy())
        
print('[INFO] Test loss: {:.6f}'.format(test_loss))
print('[INFO] Test accuracy: {:.2f}%'.format(correct/total_data * 100))

if args['save_path']:
    print('[INFO] Saving model:', args['save_path'])
    torch.save(model.state_dict(), args['save_path'])
    print('[INFO] Model has been saved')
    
print('\nModel training completed!')    

