import torch.nn as nn
import torch.nn.functional as F

class Biased_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Biased_Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, 200, bias=True)
        self.fc3 = nn.Linear(200, 100, bias=True)
        self.fc4 = nn.Linear(100, 2, bias=True)

    def forward(self, input):
        '''

        :param input: (batch_size x input_size)
        :return: (batch_size x 2)
        '''

        # x: (batch_size x input_size)
        x = input

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = self.fc4(x)

        return y

class Unbiased_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Unbiased_Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, 100, bias=True)
        self.fc3 = nn.Linear(100, 50, bias=True)
        self.fc4 = nn.Linear(50, 2, bias=True)

    def forward(self, input):
        '''

        :param input: (batch_size x input_size)
        :return: (batch_size x 2)
        '''

        # x: (batch_size x input_size)
        x = input

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = self.fc4(x)

        return y