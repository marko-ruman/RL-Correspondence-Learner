import numpy as np
from src.models.dqn.dqn_agent import DQNAgent
from src.utils.utils import make_env
from src.utils.logger import Logger


def rotate_image(image):
    # print(image.shape)
    new_image = np.zeros_like(image)
    new_image[0] = np.rot90(image[2], k=-1)
    new_image[1] = np.rot90(image[0], k=-1)
    new_image[2] = np.rot90(image[3], k=-1)
    new_image[3] = np.rot90(image[1], k=-1)
    return new_image


# def transform_observation(observation):
#     if rotate:
#         observation = rotate_image(observation)
#     state = T.tensor([observation], dtype=T.float).to(agent.q_online.device)
#     # if Gab is not None:
#     #     state = agent.square_stack(Gab(agent.square_concat_orig(state)))
#     # print(state.shape)
#     return state[0].detach().cpu().numpy()


class DQN:
    def __init__(self, env, params, model_dir, run_name, n_games=10000, mem_size=10000):

        self.env = env
        self.params = params
        self.run_name = run_name
        self.mem_size = mem_size
        self.model_dir = model_dir

        # filename = "DQN"+"_"+env_name+"_"+str(time())
        log_name = "rotated_pong_with_correct_rotation"
        filename = "q"

        epsilon_init = 0.9

        if params.get("epsilon_init", False):
            epsilon_init = params["epsilon_init"]

        elif params.get("q_filename", False):
            epsilon_init = 0.1

        self.n_games = n_games
        self.agent = DQNAgent(gamma=0.99, epsilon=epsilon_init, lr=0.0001,
                              input_dims=self.env.observation_space.shape,
                              n_actions=self.env.action_space.n, mem_size=mem_size, eps_min=0.05,
                              batch_size=32, replace=1000, eps_dec=1e-5, input_dir=self.model_dir)
        self.logger = Logger(self.agent, dir_memory="data/input")

        if params.get("q_filename", False):
            self.agent.load_q_function(model_dir+"/"+params["q_filename"])

        # image_channels = 4
        # if args.square_concat:
        #     image_channels = 1
        # Gab = None
        #
        # Gab, A, b = define_Gen(input_nc=image_channels, output_nc=image_channels, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
        #                        use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids, A=np.pi/2)
        #
        # Gab.load_state_dict(torch.load("memory/{}".format(gab_filename), map_location=agent.q_target.device))

        # for param in Gab.parameters():
        #     param.requires_grad = False

    def run(self):

        best_score = -np.inf

        # fname = self.agent.algo + '_' + self.agent.env_name + '_lr' + str(self.agent.lr) +'_' \
        #         + str(self.n_games) + 'games'
        # figure_file = 'plots/' + fname + '.png'

        n_steps = 0
        scores, eps_history, steps_array = [], [], []

        for i in range(self.n_games):
            done = False
            observation = self.env.reset()
            # observation = transform_observation(observation)
            score = 0
            while not done:
                if self.params.get("random_actions", False):
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.choose_action(observation)

                observation_, reward, done, _, info = self.env.step(action)
                score += reward

                self.agent.store_transition(observation, action, reward, observation_, int(done))
                if self.params.get("learn", False):
                    self.agent.learn()
                observation = observation_
                n_steps += 1
            scores.append(score)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print('episode: ', i, 'score: ', score,
                  ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                  'epsilon %.2f' % self.agent.epsilon, 'steps', n_steps)

            if avg_score > best_score:
                if self.params.get("save_models", False):
                    self.agent.save_q_function(self.run_name+"_q")
                # if not load_checkpoint:
                #    agent.save_models()
                best_score = avg_score

            if self.params.get("save_replay_memory_random", False) and n_steps > self.mem_size:
                self.logger.save_replay_memory(self.agent.memory, self.run_name+"_memory_random")
                break

            # if logg:
            #     logger.save_score(score)

            eps_history.append(self.agent.epsilon)
            # if load_checkpoint and n_steps >= 18000:
            #     break

        if self.params.get("save_replay_memory_final", False):
            self.logger.save_replay_memory(self.agent.memory, self.run_name+"_memory_final")

        x = [i + 1 for i in range(len(scores))]
        # plot_learning_curve(steps_array, scores, eps_history, figure_file)