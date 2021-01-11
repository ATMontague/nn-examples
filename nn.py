import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #self.fc1 = nn.Linear(768, 256)
        self.fc1 = nn.Linear(150528, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        """
        Forward portion of the net
        :param x:
        :return:
        """

        # flatten image tensor
        x = x.view(x.shape[0], -1)

        # forward pass through the net
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


def train(model, train_loader, test_loader, criterion, optimizer):
    """
    all the fun stuff

    :return:
    """

    epochs = 2
    train_losses = []
    test_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()

            # forward
            output = model(images)

            # backward
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0

            # turn off grads for better speed
            with torch.no_grad():
                for images, labels in test_loader:
                    output = model(images)
                    test_loss += criterion(output, labels)

                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
                  "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))


def load_data():
    img_dir = 'Cat_Dog_data'


    # todo: figure out why DatLoader fails when resize imgs to 16
    # train_transforms = transforms.Compose([transforms.RandomRotation(30),
    #                                        transforms.RandomResizedCrop(16),
    #                                        transforms.RandomHorizontalFlip(),
    #                                        transforms.ToTensor()])
    #
    # test_transforms = transforms.Compose([transforms.Resize(16),
    #                                       transforms.ToTensor()])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    train_data = datasets.ImageFolder(img_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(img_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)
    return trainloader, testloader


if __name__ == '__main__':
    # load data
    train_loader, test_loader = load_data()

    # make model, define loss function and optimizer
    model = Network()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    # train
    train(model, train_loader, test_loader, criterion, optimizer)
    # todo: plot training loss and validation loss
