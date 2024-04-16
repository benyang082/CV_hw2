import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
         # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected (linear) layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 4x4 image dimension after 3 max pooling layers
        self.fc2 = nn.Linear(128, 10)          # Output layer with 10 classes

        # Define activation function
        self.relu = nn.ReLU()
        

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
         # Apply first convolution, activation, and pooling
        x = self.pool(self.relu(self.conv1(x)))
        
        # Apply second convolution, activation, and pooling
        x = self.pool(self.relu(self.conv2(x)))
        
        # Apply third convolution, activation, and pooling
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 4 * 4)  # Flatten to a vector of size 64*4*4

        # Apply fully connected layers with activation
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        

    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
