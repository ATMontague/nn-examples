import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv1: sees (32,32,3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # conv2: sees (16,16,16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # conv3: sees (8,8,32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # max pooling layer, doing this after each conv layer
        self.pool = nn.MaxPool2d(2, 2)

        # linear layer
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 10)

        # drop out....i thought no dropout in CNN?
        #     we're doing dropout in the FC layers
        self.dropout = nn.Dropout(0.25)

        # use CUDA if available
        self.train_on_gpu = torch.cuda.is_available()

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        #    ie. conv -> pool -> conv -> pool ...
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten for passing to fully connected layer
        x = x.view(-1, 1024)

        # FC1 & FC2
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # why not relu?

        return x


def load_data():
    img_dir = 'Cat_Dog_data'

    # number of subprocesses to use for data loading
    num_workers = 0
    # percentage of training set to use as validation
    valid_size = 0.2
    # number of batches to load
    num_batches = 25

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    train_data = datasets.ImageFolder(img_dir + '/train', transform=transform)
    test_data = datasets.ImageFolder(img_dir + '/test', transform=transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_data, batch_size=num_batches,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(train_data, batch_size=num_batches,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=num_batches,
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def train(model, train_loader, valid_loader, test_loader, criterion, optimizer):
    """
    all the fun stuff

    :return:
    """

    epochs = 10

    # track change in validation loss
    valid_loss_min = np.Inf

    for e in range(epochs):

        # track training & validation loss
        train_loss = 0.0
        validation_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for images, labels in train_loader:
            # use CUDA if possible
            if model.train_on_gpu:
                images, labels = images.cuda(), labels.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward
            output = model(images)

            # backward
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for images, labels in valid_loader:
            # CUDA again...
            if model.train_on_gpu:
                images, labels = images.cuda(), labels.cuda()

            # forward
            output = model(images)

            # backward
            loss = criterion(output, labels)
            validation_loss += loss.item() * images.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        validation_loss = validation_loss / len(valid_loader.sampler)

        # save model if validation loss has decreased
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                validation_loss))
            torch.save(model.state_dict(), 'best_model.pt')
            valid_loss_min = validation_loss


def test(model, test_loader, classes):
    """
    testing the best model we got
    :return:
    """

    batch_s = 25

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    model.eval()
    for images, labels in test_loader:
        # forward
        output = model(images)

        # backward
        loss = criterion(output, labels)
        test_loss += loss.item() * images.size(0)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())

        # calculate test accuracy for each object class
        for i in range(batch_s):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    # load data
    train_loader, valid_loader, test_loader = load_data()
    classes = ['cat', 'dog']

    # make model, define loss function and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # train
    train(model, train_loader, valid_loader, test_loader, criterion, optimizer)
    # todo: plot training loss and validation loss

    # load the best model
    model.load_state_dict(torch.load('best_model.pt'))
    test(model, test_loader, classes)
