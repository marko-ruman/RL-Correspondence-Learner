import torch
import torchvision
import os
import numpy as np
import pickle
import json
from src.models.environment_model import EnvironmentModel
from src.models.q_function import QFunction
from src.models.correspondence_function import CorrespondenceFunction
from src.models.correspondence_function_arch import set_grad
import src.utils.utils as utils
import itertools
import time
from torch import nn
from src.utils.logger import Logger


class RLCorrespondenceLearner:

    def __init__(self, source_memory_filename: str, target_memory_filename: str, environment,
                 correspondence_function_args, input_dir, log_dir, weights=None,
                 environment_model_filename: str = None, q_filename: str = None, run_name=str(utils.current_milli_time())):

        self.input_dir = input_dir

        self.log_dir = log_dir+"/"+run_name+"/q_"+str(weights.get("q", 0))+"m_"+str(weights["model"])+"c_"\
                       + str(weights["cycle"])

        os.makedirs(self.log_dir, exist_ok=True)

        if weights is None:
            weights = {"q": 10, "model": 10, "cycle": 10}
        self.weights = weights

        self.correspondence_function_args = correspondence_function_args

        self.correspondence_function_args.cycle_weight = self.weights["cycle"]
        self.correspondence_function_args.q_weight = self.weights.get("q",  0)
        self.correspondence_function_args.model_weight = self.weights["model"]

        with open(self.log_dir+'/params.txt', 'w') as file:
            file.write(json.dumps({**weights, **vars(self.correspondence_function_args)}))

        self.logger = Logger(dir_results=self.log_dir)

        self.source_memory_filename = source_memory_filename
        self.target_memory_filename = target_memory_filename

        self.source_memory = []
        self.target_memory = []

        self.env = environment

        self.q_function = None
        if q_filename is not None:
            self.q_function = QFunction(0, self.env.action_space.n,
                                        input_dims=self.env.observation_space.shape,
                                        input_dir=self.input_dir)
            self.q_function.load_q(q_filename, True)

        self.model = None
        if environment_model_filename is not None:
            print(self.env.action_space.n, self.env.observation_space.shape)
            opt = {
                "action_dim": self.env.action_space.n
            }
            self.environment_model = EnvironmentModel(opt).cuda()
            self.environment_model.load_state_dict(torch.load(f"{self.input_dir}/{environment_model_filename}",
                                                              map_location=self.environment_model.device))

        self.correspondence_function = CorrespondenceFunction(self.correspondence_function_args,
                                                              q_function=self.q_function,
                                                              environment_model=self.environment_model)
        # Define Loss criteria

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.kl_loss = nn.KLDivLoss(reduction="sum", log_target=True)

        self.g_optimizer = torch.optim.Adam(itertools.chain(self.correspondence_function.Gab.parameters(),
                                                            self.correspondence_function.Gba.parameters()),
                                            lr=self.correspondence_function_args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.correspondence_function.Da.parameters(),
                                                            self.correspondence_function.Db.parameters()),
                                            lr=self.correspondence_function_args.lr, betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(
                                                                    self.correspondence_function_args.epochs, 0,
                                                                    self.correspondence_function_args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(
                                                                    self.correspondence_function_args.epochs, 0,
                                                                    self.correspondence_function_args.decay_epoch).step)

    def save_corresponding_image_states(self, a_real, b_real, filename):

        self.correspondence_function.Gab.eval()
        self.correspondence_function.Gba.eval()

        with torch.no_grad():
            a_fake = self.correspondence_function.Gab(b_real)
            b_fake = self.correspondence_function.Gba(a_real)
            a_recon = self.correspondence_function.Gab(b_fake)
            b_recon = self.correspondence_function.Gba(a_fake)

        if not self.correspondence_function_args.square_concat:
            a_real = utils.square_concat(a_real)
            b_real = utils.square_concat(b_real)
            a_fake = utils.square_concat(a_fake)
            b_fake = utils.square_concat(b_fake)
            a_recon = utils.square_concat(a_recon)
            b_recon = utils.square_concat(b_recon)

        pic = (torch.cat([a_real, b_fake, a_recon, b_real, a_fake, b_recon],
                         dim=0).data + 1) / 2.0

        torchvision.utils.save_image(pic, self.log_dir + '/{}.jpg'.format(filename), nrow=3)

        self.correspondence_function.Gab.train()
        self.correspondence_function.Gba.train()

    @staticmethod
    def rotate(image):
        new_image = np.zeros_like(image)
        new_image[0] = np.rot90(image[2], k=-1)
        new_image[1] = np.rot90(image[0], k=-1)
        new_image[2] = np.rot90(image[3], k=-1)
        new_image[3] = np.rot90(image[1], k=-1)
        return new_image

    def save_single_image(self, image, name):
        def square_concat(image):
            c = np.concatenate(image[:int(image.shape[0] / 2)], axis=1)
            d = np.concatenate([c, np.concatenate(image[int(image.shape[0] / 2):], axis=1)])
            return d

        dir = self.log_dir + f"/single_images/"
        os.makedirs(dir, exist_ok=True)
        pic = square_concat(image)
        pic = torch.tensor(pic).data
        torchvision.utils.save_image(pic, dir + '/{}.jpg'.format(name), nrow=3)

    def test_play(self):

        n_games = 5
        scores = []
        average_reward = 0.0
        j = 0
        for i in range(n_games):
            # done = False
            done = False
            observation, _ = self.env.reset()

            score = 0

            while not done:
                j += 1
                with torch.no_grad():
                    # state = T.tensor([observation], dtype=T.float).to(self.Q.device)

                    # transformed_state = state
                    # transformed_state = self.rotate(transformed_state)

                    if self.q_function is not None:
                        if self.correspondence_function_args.rotate:
                            state = torch.tensor([self.rotate(observation)], dtype=torch.float).cuda()
                        else:
                            state = torch.tensor([observation], dtype=torch.float).cuda()
                        transformed_state = utils.square_stack(self.correspondence_function.Gab(
                            utils.square_concat(state, True)))
                        actions = self.q_function.forward(transformed_state)
                        action = torch.argmax(actions).item()

                observation_, reward, done, _, info = self.env.step(action)
                score += reward

                observation = observation_
            scores.append(score)
            average_reward += score

            avg_score = np.mean(scores[-100:])
            print('episode: ', i, 'score: ', score,
                  ' average score %.1f' % avg_score)

        return average_reward/n_games

    def learn(self):

        source_memory = pickle.load(open(self.input_dir + "/" + self.source_memory_filename, "rb"))
        target_memory = pickle.load(open(self.input_dir + "/" + self.target_memory_filename, "rb"))

        max_average_reward = -np.inf

        a_loader = torch.utils.data.DataLoader(source_memory, batch_size=self.correspondence_function_args.batch_size,
                                               shuffle=True)
        b_loader = torch.utils.data.DataLoader(target_memory, batch_size=self.correspondence_function_args.batch_size,
                                               shuffle=True)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.correspondence_function_args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a, b) in enumerate(zip(a_loader, b_loader)):
                # print(type(a[0]))
                # sleep(100)
                a_real = a[0]
                a_action = a[1]
                a_real_next = a[2]
                b_real = b[0]
                b_action = b[1]
                b_real_next = b[2]

                if self.correspondence_function_args.square_concat:
                    a_real = utils.square_concat(a_real, True)
                    # a_real = a_real.reshape((1, ) + tuple(a_real.shape[:]))
                    a_real_next = utils.square_concat(a_real_next, True)
                    # a_real_next = a_real_next.reshape((1, ) + tuple(a_real_next.shape[:]))
                    b_real = utils.square_concat(b_real, True)
                    b_real_next = utils.square_concat(b_real_next, True)
                    # b_real = b_real.reshape((1, ) + tuple(b_real.shape[:]))
                # else:
                #     a_real = a_real*2.0 - 1.0
                #     a_real_next = a_real_next*2.0 - 1.0
                #     b_real = b_real*2.0 - 1.0

                # step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1
                # Generator Computations
                ##################################################

                set_grad([self.correspondence_function.Da, self.correspondence_function.Db], False)
                self.g_optimizer.zero_grad()

                # a_real = Variable(a_real[0])
                # b_real = Variable(b_real[0])
                a_real, b_real, a_real_next, a_action = utils.cuda([a_real, b_real, a_real_next, a_action])

                # Forward pass through generators
                ##################################################
                a_fake = self.correspondence_function.Gab(b_real)
                b_fake = self.correspondence_function.Gba(a_real)

                a_recon = self.correspondence_function.Gab(b_fake)
                b_recon = self.correspondence_function.Gba(a_fake)

                if self.q_function is not None:
                    set_grad([self.q_function], False)
                if self.environment_model is not None:
                    set_grad([self.environment_model], False)

                # Adversarial losses
                ###################################################
                # print(a_fake.size(), a_real.size())
                # sleep(100)
                a_fake_dis = self.correspondence_function.Da(a_fake)
                b_fake_dis = self.correspondence_function.Db(b_fake)

                real_label = utils.cuda(torch.ones(a_fake_dis.size()))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * self.correspondence_function_args.cycle_weight
                b_cycle_loss = self.L1(b_recon, b_real) * self.correspondence_function_args.cycle_weight

                # Q loss
                ###################################################
                if self.correspondence_function_args.q_weight > 0:
                    Q_r = self.q_function(utils.square_stack(a_recon))
                    Q = self.q_function(utils.square_stack(a_real))
                    a_q_loss = (self.L1(Q_r, Q)) * self.correspondence_function_args.q_weight

                # Model loss
                ###################################################
                if self.correspondence_function_args.model_weight > 0:
                    a_fake_next = self.correspondence_function.Gab(b_real_next)
                    a_fake_next_model = self.environment_model(utils.square_stack(a_fake), b_action)

                    if self.correspondence_function_args.model_l1:
                        a_model_loss = (self.L1(utils.square_stack(a_fake_next), a_fake_next_model)) \
                                       * self.correspondence_function_args.model_weight
                    else:
                        a_model_loss = (self.MSE(utils.square_stack(a_fake_next), a_fake_next_model)) \
                                       * self.correspondence_function_args.model_weight

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss

                if self.correspondence_function_args.q_weight > 0:
                    gen_loss = gen_loss + a_q_loss

                if self.correspondence_function_args.model_weight > 0:
                    gen_loss = gen_loss + a_model_loss


                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()


                # Discriminator Computations
                #################################################

                set_grad([self.correspondence_function.Da, self.correspondence_function.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0])
                b_fake = torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0])
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                #################################################

                a_real_dis = self.correspondence_function.Da(a_real)
                a_fake_dis = self.correspondence_function.Da(a_fake)
                b_real_dis = self.correspondence_function.Db(b_real)
                b_fake_dis = self.correspondence_function.Db(b_fake)

                real_label = utils.cuda(torch.ones(a_real_dis.size()))
                fake_label = utils.cuda(torch.zeros(a_fake_dis.size()))

                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss+b_dis_loss))

                errors = {"a_dis_loss": a_dis_loss.item(), "b_dis_loss": b_dis_loss.item(),
                          "a_gen_loss": a_gen_loss.item(), "b_gen_loss": b_gen_loss.item()}

                if self.correspondence_function_args.cycle_weight > 0:
                    errors["a_cycle_loss"] = a_cycle_loss.item()/self.correspondence_function_args.cycle_weight
                    errors["b_cycle_loss"] = b_cycle_loss.item()/self.correspondence_function_args.cycle_weight
                if self.correspondence_function_args.q_weight > 0:
                    errors["a_q_loss"] = a_q_loss.item()/self.correspondence_function_args.q_weight
                if self.correspondence_function_args.model_weight > 0:
                    errors["a_model_loss"] = a_model_loss.item()/self.correspondence_function_args.model_weight


                # save errors to csv
                self.logger.save_errors(errors)

                if i % 300 == 0:
                    self.save_corresponding_image_states(a_real, b_real, str(i+epoch*min(len(a_loader), len(b_loader))))

                if i % 1000 == 0:
                    average_reward_test = self.test_play()
                    errors = {"average_reward": average_reward_test,
                              "step_play": i+epoch*min(len(a_loader), len(b_loader))}
                    self.logger.save_errors(errors)
                    if average_reward_test > max_average_reward:
                        max_average_reward = average_reward_test
                        torch.save(self.correspondence_function.Gab.state_dict(), self.log_dir + '/{}'
                                   .format("best_model"))

            average_reward_test = self.test_play()

            errors = {"average_reward": average_reward_test, "step_play": i+epoch*min(len(a_loader), len(b_loader))}
            self.logger.save_errors(errors)
            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()