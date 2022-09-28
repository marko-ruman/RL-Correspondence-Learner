from src.models.rl_correspondence_learner import RLCorrespondenceLearner
from src.utils.utils import make_env
from argparse import ArgumentParser
from multiprocessing import Pool


def get_args():
    parser = ArgumentParser(description='RL Correspondence Learner')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--decay_epoch', type=int, default=3)
    parser.add_argument('--noise_finish', type=int, default=4)
    parser.add_argument('--noise_init_std', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--lr_affine', type=float, default=.0002)
    parser.add_argument('--load_height', type=int, default=286)
    parser.add_argument('--load_width', type=int, default=286)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=256)
    parser.add_argument('--crop_width', type=int, default=256)
    parser.add_argument('--gen_lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0)
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_6blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    parser.add_argument('--dis_net_layers', type=int, default=3)
    parser.add_argument('--square_concat', type=bool, default=True)
    parser.add_argument('--lower_true_label', type=bool, default=False)
    parser.add_argument('--model_l1', type=bool, default=True)
    parser.add_argument('--rotate', type=bool, default=True)

    args = parser.parse_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    return args


data_input_dir = "data/input"
log_dir = "data/log"

source_memory_filename = "pong"
target_memory_filename = "pong_difficulty_3"

model_filename = "model_pong"

q_filename = "q_pong"

environment_name = "PongNoFrameskip-v4"
environment = make_env(environment_name, difficulty=3)

# weights = {"q": 1, "model": 1, "cycle": 10}

num_processes = 1

# weights_dict = {"q": [0, 1], "model": [0, 1, 10], "cycle": [0, 10, 100]}
weights_dict = {"q": [0], "model": [1], "cycle": [100]}

weights_list = []

for weight_q in weights_dict["q"]:
    for weight_model in weights_dict["model"]:
        for weight_cycle in weights_dict["cycle"]:
            weights_list.append({"q": weight_q, "model": weight_model, "cycle": weight_cycle})


def learn_for_parameters(weights):
    learner = RLCorrespondenceLearner(source_memory_filename=source_memory_filename,
                                      target_memory_filename=target_memory_filename, input_dir=data_input_dir,
                                      log_dir=log_dir, environment=environment,
                                      environment_model_filename=model_filename,
                                      q_filename=q_filename, weights=weights, correspondence_function_args=args)

    learner.learn()


if __name__ == '__main__':
    args = get_args()

    if len(weights_list) > 1 and num_processes > 1:
        with Pool(num_processes) as p:
            p.map(learn_for_parameters, weights_list)
    else:
        learn_for_parameters(weights_list[0])


