import numpy as np
import torch as T
from src.models.q_function import QFunction
from src.models.dqn.replay_buffer import ReplayBuffer

import src.utils.utils as utils
import torch


class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, input_dir="data/input"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_online = QFunction(self.lr, self.n_actions, input_dims=self.input_dims, input_dir=input_dir)

        self.q_target = QFunction(self.lr, self.n_actions, input_dims=self.input_dims, input_dir=input_dir)

    def square_concat(self, image):
        image = image.detach().cpu().numpy()[0]
        c = np.concatenate(image[:int(image.shape[0]/2)], axis=1)
        d = np.concatenate([c, np.concatenate(image[int(image.shape[0]/2):], axis=1)])
        # d = d*2.0-1
        return torch.tensor(d)

    def square_concat_orig(self, image):
        image = image.detach().cpu().numpy()[0]
        c = np.concatenate(image[:int(image.shape[0]/2)], axis=1)
        d = np.concatenate([c, np.concatenate(image[int(image.shape[0]/2):], axis=1)])
        d = d*2.0-1
        d = d.reshape((1, 1)+ tuple(d.shape[:]))
        return utils.cuda(torch.tensor(d))

    def square_stack(self, image):
        image = image[0, 0]
        w = int(image.shape[0] / 2)
        h = int(image.shape[1] / 2)
        e = utils.cuda(torch.zeros((4, w, h)))
        e[0] = image[:w, :h]
        e[1] = image[:w, h:]
        e[2] = image[w:, :h]
        e[3] = image[w:, h:]
        e = (e+1.0)/2.0
        e = e.reshape((1, )+tuple(e.shape))
        return utils.cuda(e)

    def choose_action(self, observation, Gab=None):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_online.device)
            if Gab is not None:
                state = self.square_stack(Gab(self.square_concat_orig(state)))
                # transformed_state = self.square_stack(self.square_concat_orig(state))
                # actions = self.Q.forward(transformed_state)
            actions = self.q_online.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_online.device)
        rewards = T.tensor(reward).to(self.q_online.device)
        dones = T.tensor(done).to(self.q_online.device)
        actions = T.tensor(action).to(self.q_online.device)
        states_ = T.tensor(new_state).to(self.q_online.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_q_function(self, q_filename):
        self.q_online.save_q(q_filename)
        self.q_target.save_q(q_filename)

    def load_q_function(self, q_filename):
        self.q_online.load_q(q_filename, False)
        self.q_target.load_q(q_filename, False)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_online.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_online.forward(states)[indices, actions]
        q_next = self.q_target.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_online.loss(q_target, q_pred).to(self.q_online.device)
        loss.backward()
        self.q_online.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
