import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

loss_array = []

def train(network, critiern, optimizer, train_batches, test_batches, epoch=10): 

    # Things needed to train a machine learning model:
    # - Run the model a sample
    # - Calc cost/loss function
    # - "Backpropigation" / some way of chanigne the wieghts

    num_epochs = epoch

    for epoch in range(num_epochs):
        # Need to put the network in training mode

        loss_train = 0
        loss_eval = 0
        running_loss = 0

        network.train()
        for step in range(len(train_batches)): 
            images, landmarks = next(iter(train_batches))

            x = images.cuda()
            # I think this flatterns out the landmarks
            y = landmarks.view(landmarks.size(0), -1).cuda()


            # This resets teh grad values the perameters 
            optimizer.zero_grad()

            pred = network(x)
            # This calculates loss based on MeanSquaredError(MSE)
            cost = critiern(pred, y)
            # Performs back propigation to train the network
            cost.backward()
            # updates the weights in the model
            optimizer.step()


            loss_train += cost.item()
            # Average loss over this epoch
            running_loss = loss_train/(step+1)
            loss_array.append(running_loss)

            print_overwrite(step, len(train_batches), loss_train, 'train')

        network.eval()
        with torch.no_grad():
            # Run over the validation dataset, this means split up the dataset
            for step in range(len(test_batches)): 
                images, landmarks = next(iter(test_batches))

                x = images.cuda()
                # I think this flatterns out the landmarks
                y = landmarks.view(landmarks.size(0), -1).cuda()
                # x = image
                # y = landmarks


                pred = network(x)
                # What does this do?
                cost_eval = critiern(pred, y)
                # print(f"\nCost: {cost_eval}")

                loss_eval += cost_eval.item()
                # Average loss over this epoch
                running_loss = loss_eval/(step+1)

                print_overwrite(step, len(test_batches), loss_eval, 'eval')
        loss_train /= len(test_batches)
        loss_eval /= len(test_batches)


        print('\n' + '-' *20)
        print('Epoch: {} Train Loss: {:.4f} Eval Loss: {:.4f}'.format(epoch, loss_train, loss_eval))
        print('\n' + '-' *20)

        # Save the net
        torch.save(network.state_dict(), 'content/face_landmarks.pth')
        print('\n Saved a new model\n')

    print("Training Complete")
    show_prediction(next(iter(train_batches)), network)



def show_prediction(data, network):
    with torch.no_grad(): 
        network.eval()

        images, landmarks = data

        images = images.cuda()
        landmarks = (landmarks + 0.5) * 224

        pred = (network(images).cpu() + 0.5) * 224
        z = pred.view(-1,68,2)


        plt.figure(figsize=(10, 40))
        plt.imshow(images[0].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray')
        plt.scatter(z[0,:, 0], z[0,:, 1], c=[[1,0,0]],  s=8)
        plt.scatter(landmarks[0,:, 0], landmarks[0,:, 1], c=[[0,1,0]], s=8)
        plt.figure(2)
        plt.scatter(loss_array, [x for x in range(1,len(loss_array)+1)], c=[[0,0,1]])
        plt.show()


def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write('Train Steps: %d/%d Running Loss: %.4f' % (step, total_step, loss))

        loss_array.append(loss)
    else:
        sys.stdout.write('Eval Steps: %d/%d Loss: %.4f ' % (step, total_step, loss))
