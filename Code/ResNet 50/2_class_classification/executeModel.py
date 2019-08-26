import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.dataloader import *
from helpers.trainModel import *
from helpers.evalModel import *
import pickle
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from inceptionResnet_v2 import *
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import datetime

print('Training started at: {}'.format(datetime.datetime.now()))

if torch.cuda.is_available:
    device = torch.device('cuda')
    print('#{} CUDA device(s) available'.format(torch.cuda.device_count()))
else:
    device = torch.device('cpu')
    print('No CUDA devices available')
print('-' * 20)

modelName = ''
root_dir = '/scratch/bva212/breastCancerData'
model_folder = 'temp_testing/'
project_save_loc = '/scratch/bva212/dl4medProject/'
project_save_dir = 'ResNet50_2_Classes/'
image_column = 'image file path'
if torch.cuda.device_count() >= 2:
    batchSize = 8
else:
    batchSize = 4
num_classes = 2
if num_classes == 2:
    target_names = ['Benign', 'Malignant']
else:
    target_names = ['Calcification - Benign', 'Calcification - Malignant', 'Mass - Benign', 'Mass - Malignant']
num_channel = 3
image_resize = 1024
transform_prob = 0.5
numEpochs = 100
dropout_rate = 0.2
learning_rate = 0.0005
verbose = True 
print_every = 3
save = True
save_every = 20
# train_file = 'randomTrainSet.pkl'
# val_file = 'randomValidationSet.pkl'
# test_file = 'randomTestSet.pkl'
train_file = 'Train.pkl'
val_file = 'Val.pkl'
test_file = 'Test.pkl'


print('Training Stats:')
print('#Classes: {} | #Image Size: {} | #Batch Size: {} | #Epochs: {} | Learning Rate: {}'.format(num_classes, image_resize, batchSize, numEpochs, learning_rate))
print('Saving model in: {}'.format(os.path.join(project_save_loc,project_save_dir)))
print('-' * 20)

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([image_resize,image_resize]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([image_resize,image_resize]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



train_data = CBISDDSMDataset(train_file, root_dir, image_column, num_classes = num_classes, num_channel=num_channel, transform = train_transform, transform_type = None, transform_prob=transform_prob)
val_data = CBISDDSMDataset(val_file, root_dir, image_column, num_classes = num_classes, num_channel=num_channel, transform = validation_transform, transform_type = None, transform_prob=transform_prob)
test_data = CBISDDSMDataset(test_file, root_dir, image_column, num_classes = num_classes, num_channel=num_channel, transform = validation_transform, transform_type = None, transform_prob=transform_prob)


image_datasets = {'train': train_data, 'val': val_data, 'test': test_data}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

model = torchvision.models.resnet50(pretrained= True)
fc_in_features = model.fc.in_features
model.fc = torch.nn.Linear(fc_in_features, num_classes)
print("Using model ResNet50 on 2 classes")
print('-' * 20)
print('')

print('Training Start')
print('-' * 20)
print('')

model = model.to(device)
modelParallel = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(modelParallel.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
BestConvNetModel = train_model(modelParallel, dataloaders, criterion, optimizer, scheduler, dataset_sizes,
                           num_epochs = numEpochs, verbose = verbose, 
                           print_every = print_every, save = save, save_every = save_every, 
                           root_dir = project_save_loc, 
                           model_folder = project_save_dir)

print('')

print('Training Completed')
print('')


print('Validation Start')
print('-' * 20)
print('')

outputs, preds, labels, accuracy, loss = evaluate_model(modelParallel, dataloaders, criterion, phase = 'test')

print('')
print('')

print('Validation End')

print('')


# plot_confusion_matrix(cm, normalize = False, target_names = target_names, 
#                       title = "Confusion Matrix of {}".format(modelName), root_dir = project_save_loc, model_folder = project_save_dir)


# fpr, tpr, _ = roc_curve(labels, preds)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label = '{} AUC Score: {:.4f}'.format(modelName, auc(fpr, tpr)))
# plt.title('')
# plt.legend()
# plt.savefig(os.path.join(project_save_loc,project_save_dir, 'AUC_Score - Test'))
# plt.show()

y_s, y_t=inference(modelParallel, dataloaders['test'])
r_AUC_wholeSet = get_AUC(y_s, y_t,True)
print('AUC: ', r_AUC_wholeSet)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_t, np.argmax(y_s,axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')


print('--------------------THE END------------------')