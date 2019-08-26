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
project_save_dir = 'temp_testing/'
image_column = 'image file path'
batchSize = 2
num_classes = 2
if num_classes == 2:
	target_names = ['Benign', 'Malignant']
else:
	target_names = ['Calcification - Benign', 'Calcification - Malignant', 'Mass - Benign', 'Mass - Malignant']
num_channel = 3
image_resize = 1024
transform_prob = 0.5
numEpochs = 3 
dropout_rate = 0.2
learning_rate = 0.0005
verbose = True 
print_every = 3
save = False
save_every = 20
train_file = 'randomTrainSet.pkl'
val_file = 'randomValidationSet.pkl'
test_file = 'randomTestSet.pkl'
# train_file = 'Train.pkl'
# val_file = 'Val.pkl'
# test_file = 'Test.pkl'


print('Training Stats:')
print('#Classes: {} | #Image Size: {} | #Batch Size: {} | #Epochs: {} | Learning Rate: {}'.format(num_classes, image_resize, batchSize, numEpochs, learning_rate))
print('Saving model & plots in: {}'.format(os.path.join(project_save_loc,project_save_dir)))
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


class ConvNetWithResidualUnits(nn.Module):
    def __init__(self):
        super(ConvNetWithResidualUnits, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.BatchNorm2d(64)
        )
        
        self.residual1_conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.residual1_bn = nn.BatchNorm2d(64)        
        self.residual1_conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2),
            nn.PReLU(256),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.BatchNorm2d(256)
        )
        
        self.residual2_conv1 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.residual2_bn = nn.BatchNorm2d(256)        
        self.residual2_conv2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=2),
            nn.PReLU(1024),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.BatchNorm2d(1024)
        )
        
        self.residual3_conv1 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1)
        self.residual3_bn = nn.BatchNorm2d(1024)
        self.residual3_conv2 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=2),
            nn.PReLU(2048),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.BatchNorm2d(2048)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, stride=2),
            nn.PReLU(4096),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.BatchNorm2d(4096)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=1024, kernel_size=3, stride=2, padding = 1),
            nn.PReLU(1024),
            nn.AdaptiveMaxPool2d(1)
        )
        
        self.out = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        
        x = self.conv1(x)
        
        residual = x
        x = self.residual1_bn(F.relu(self.residual1_conv1(x)))
        x = self.residual1_conv2(x)
        x += residual
        x = F.relu(x)
        
        x = self.conv2(x)
        
        residual = x
        x = self.residual2_bn(F.relu(self.residual2_conv1(x)))
        x = self.residual2_conv2(x)
        x += residual
        x = F.relu(x)
        
        x = self.conv3(x)
        
        residual = x
        x = self.residual3_bn(F.relu(self.residual3_conv1(x)))
        x = self.residual3_conv2(x)
        x += residual
        x = F.relu(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)
        
        x = self.conv6(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.out(x))
        
        return x

ConvNetModel = ConvNetWithResidualUnits()
print(ConvNetModel)

print('')
print('')
print('')

print('Training Start')
print('-' * 20)

print('')
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
print('')
print('')

print('Training Completed')

print('')
print('')
print('')

print('Validation Start')
print('-' * 20)

print('')
print('')
print('')

outputs, preds, labels, accuracy, loss = evaluate_model(BestConvNetModel, dataloaders, criterion, phase = 'test')
cm  = confusion_matrix(labels, preds)

print('')
print('')
print('')

print('Validation End')

print('')
print('')
print('')


plot_confusion_matrix(cm, normalize = False, target_names = target_names, 
                      title = "Confusion Matrix of {}".format(modelName), root_dir = project_save_loc, model_folder = project_save_dir)

fig = plt.figure()
fpr, tpr, _ = roc_curve(labels, preds)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label = '{} AUC Score: {:.4f}'.format(modelName, auc(fpr, tpr)))
plt.title('')
plt.legend()
plt.savefig(os.path.join(project_save_loc,project_save_dir, 'AUC_Score - Test'))
plt.show()




