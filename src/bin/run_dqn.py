from src.models.dqn.DQN import DQN
from src.utils.utils import current_milli_time, make_env

environment_name = "PongNoFrameskip-v4"

environment = make_env(environment_name, difficulty=3)

params = {"save_replay_memory_random": True, "random_actions": True}

run_name = environment_name+"_"+str(current_milli_time())

dqn = DQN(env=environment, params=params, model_dir="data/input", run_name=run_name)

dqn.run()
