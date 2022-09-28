import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class QFunction(nn.Module):
    def __init__(self, lr, n_actions, input_dims, input_dir):
        super(QFunction, self).__init__()
        self.input_dir = input_dir

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def load_q(self, q_filename, turn_off_grad=False):
        print('... loading Q ...')
        q_filename = os.path.join(self.input_dir, q_filename)
        Q = T.load(q_filename)

        self.load_state_dict(Q)
        if turn_off_grad:
            for child in self.children():
                for param in child.parameters():
                    param.requires_grad = False

    def save_q(self, q_filename):
        print('... saving Q ...')
        q_filename = os.path.join(self.input_dir, q_filename)
        T.save(self.state_dict(), q_filename)



