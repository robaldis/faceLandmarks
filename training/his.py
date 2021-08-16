import numpy as np
import time
import torch
import sys

loss_array = []

def train(network, criterion, optimizer, train_loader, valid_loader, epoch=10):
    network.cuda()    

    loss_min = np.inf
    num_epochs = epoch

    start_time = time.time()
    for epoch in range(1,num_epochs+1):
        
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        
        network.train()
        for step in range(1,len(train_loader)+1):
        
            images, landmarks = next(iter(train_loader))
            
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
            
            predictions = network(images)
            
            # clear all the gradients before calculating them
            optimizer.zero_grad()
            
            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)
            
            # calculate the gradients
            loss_train_step.backward()
            
            # update the parameters
            optimizer.step()
            
            loss_train += loss_train_step.item()
            running_loss = loss_train/step
            
            print_overwrite(step, len(train_loader), running_loss, 'train')
            
        network.eval() 
        with torch.no_grad():
            
            for step in range(1,len(valid_loader)+1):
                
                images, landmarks = next(iter(valid_loader))
            
                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0),-1).cuda()
            
                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')
        
        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)
        
        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')
        
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), 'content/face_landmarks.pth') 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')
         
    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write('Train Steps: %d/%d Running Loss: %.4f' % (step, total_step, loss))

        loss_array.append(loss)
    else:
        sys.stdout.write('Eval Steps: %d/%d Loss: %.4f ' % (step, total_step, loss))
