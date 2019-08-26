import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.dataloader import *
import pickle
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes,
                num_epochs = 10,verbose = True, print_every = 1, save = True, save_every = 20, 
                root_dir = '/scratch/bva212/dl4medProject/', model_folder = ''):

    if os.path.exists(os.path.join(root_dir,model_folder)):
        
        start_time = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        acc_dict = {'train':[],'validation':[]}
        # auc_dict = {'train':[],'validation':[]}
        loss_dict = {'train':[],'validation':[]}

        for epoch in range(num_epochs):

            epoch_start_time = time.time()

            if verbose:
                if epoch % print_every == 0:
                    print('Epoch {}/{}'.format(epoch+1, num_epochs))
                    print('-' * 10)

            for phase in ['train','val']:
                if phase == 'train':
                    model.train(True)
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for data in dataloaders[phase]:

                    inputs = data['x']
                    labels = data['y']

                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.type(torch.LongTensor).to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # predictions += list(preds.numpy())
                        # truths += list(labels.numpy())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size()[0]
                    running_corrects += torch.sum(preds == labels).item()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                # epoch_auc = roc_auc_score(truths, predictions)

                if verbose:
                    if epoch % print_every == 0:
                        # print('{} Loss: {:.4f} Acc: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auc))
                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if save:
                    if epoch % save_every == 0:
                        if os.path.exists(os.path.join(root_dir,model_folder, 'modelStateDict.pt')):
                            os.remove(os.path.join(root_dir,model_folder, 'modelStateDict.pt'))
                        if os.path.exists(os.path.join(root_dir,model_folder, 'optimStateDict.pt')):
                            os.remove(os.path.join(root_dir,model_folder, 'optimStateDict.pt'))
                        torch.save(model.state_dict(), os.path.join(root_dir,model_folder, 'modelStateDict.pt'))
                        torch.save(optimizer.state_dict(), os.path.join(root_dir,model_folder, 'optimStateDict.pt'))

                if phase == 'train':
                    loss_dict['train'].append(epoch_loss)
                    acc_dict['train'].append(epoch_acc)
                    # auc_dict['train'].append(epoch_auc)
                else:
                    loss_dict['validation'].append(epoch_loss)
                    acc_dict['validation'].append(epoch_acc)
                    # auc_dict['validation'].append(epoch_auc)
                    scheduler.step(epoch_loss)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            if epoch % print_every == 0:
                print('')

        time_elapsed = time.time() - start_time
        print('Training time: {}minutes {}s'.format(int(time_elapsed / 60), time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        for i, phase in enumerate(['train','validation']):

            fig = plt.figure()#figsize = (15, 12))
            plt.plot(loss_dict[phase])
            plt.title('Loss per epoch for ' + phase)
            plt.legend()
            plt.savefig(os.path.join(root_dir,model_folder, 'EpochWiseLoss_' + phase))
            plt.show()


            fig = plt.figure()
            plt.plot(acc_dict[phase])
            plt.title('Accuracy per epoch for ' + phase)
            plt.legend()
            plt.savefig(os.path.join(root_dir,model_folder, 'EpochWiseAccuracy_' + phase))
            plt.show()

        model.load_state_dict(best_model_wts)

        if save:
            if os.path.exists(os.path.join(root_dir,model_folder, 'bestModelStateDict.pt')):
                os.remove(os.path.join(root_dir,model_folder, 'bestModelStateDict.pt'))
            if os.path.exists(os.path.join(root_dir,model_folder, 'bestOptimStateDict.pt')):
                os.remove(os.path.exists(os.path.join(root_dir,model_folder, 'bestOptimStateDict.pt')))
            torch.save(model.state_dict(), os.path.join(root_dir,model_folder, 'bestModelStateDict.pt'))
            torch.save(optimizer.state_dict(), os.path.join(root_dir,model_folder, 'bestOptimStateDict.pt'))
        return model
    else: 
        print("Directory doesn't exist")