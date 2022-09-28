import copy
import os
import shutil
import time
import numpy as np
import torch

def current_milli_time():
    return round(time.time() * 1000)

# To make directories 
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]

# For Pytorch data loader
def create_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], 'Link'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], 'Link'))

    return dirs

def get_traindata_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    return dirs

def get_testdata_link(dataset_dir):
    dirs = {}
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    return dirs


# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')


def square_concat(image, reshape=False):
    image = image.detach().cpu().numpy()[0]
    c = np.concatenate(image[:int(image.shape[0]/2)], axis=1)
    d = np.concatenate([c, np.concatenate(image[int(image.shape[0]/2):], axis=1)])
    d = d*2.0-1
    if reshape:
        d = d.reshape((1, 1) + tuple(d.shape[:]))
    return cuda(torch.tensor(d))


def square_stack(image):
    image = image[0, 0]
    w = int(image.shape[0] / 2)
    h = int(image.shape[1] / 2)
    e = cuda(torch.zeros((4, w, h)))
    e[0] = image[:w, :h]
    e[1] = image[:w, h:]
    e[2] = image[w:, :h]
    e[3] = image[w:, h:]
    e = (e+1.0)/2.0
    e = e.reshape((1, )+tuple(e.shape))
    return cuda(e)


import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
# import retro
import os
# from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
# from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# retro.data.Integrations.add_custom_path(
#     os.path.join(SCRIPT_DIR, "custom_integrations")
# )


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        # print((2,) + self.shape)
        self.frame_buffer = np.zeros((2,) + self.shape)
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, _, info = self.env.step(action)
            # obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, _, info
        # return max_frame, t_reward, done, info

    def reset(self):
        obs, _ = self.env.reset()
        # obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros((2,) + self.shape, dtype=np.uint8)
        self.frame_buffer[0] = obs

        # return obs
        return obs, None


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        # print(obs)
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        # print(self.env.reset())
        observation, _ = self.env.reset()
        # observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False, difficulty=0, mode=0, render_mode='rgb_array'):
    env = gym.make(env_name, difficulty=difficulty, mode=mode, render_mode=render_mode)
    # env = gym.make(env_name, difficulty=difficulty, mode=mode)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    # env = RepeatActionAndMaxFrame(env, 20, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env

#
# def make_env_one_retro(rank):
#     """
#     creates environment
#     :param rank: # of the current process
#     :param use_gan: True if using gan for translation, False otherwise
#     :param which_epoch: the epoch the model was saved
#     :param arguments: arguments received from the user
#     """
#     def _thunk():
#         env = retro.make(game='RoadFighter-Nes', state='RoadFighter.Level{}'.format(3),
#                          inttype=retro.data.Integrations.ALL)
#
#         env.seed(1 + rank)
#
#         env = WarpFrame(env)
#         # env = MaxAndSkipEnv(env, skip=4)
#
#         return env
#
#     return _thunk
#
#
# def make_env_retro():
#
#     env = [make_env_one_retro(i) for i in range(1)]
#
#     env = DummyVecEnv(env)
#     env = VecFrameStack(env, n_stack=4)
#     env = VecNormalize(env, norm_reward=False)
#
#     return env