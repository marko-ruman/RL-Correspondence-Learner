import gym
from src.utils.utils import make_env

# env = gym.make("PongNoFrameskip-v4", render_mode='human', mode=0, difficulty=3)
env = make_env("PongNoFrameskip-v4", mode=0, difficulty=3, render_mode='human')
# env.mode = 'human'
n_games = 1

for i in range(n_games):
    done = False
    o = env.reset()
    # observation = transform_observation(observation)
    score = 0
    while not done:
        # action = agent.choose_action(observation, Gab)
        action = env.action_space.sample()

        env.step(action)
        env.render()
        # score += reward
        # observation_ = transform_observation(observation_)
        # if not load_checkpoint:
        # agent.store_transition(observation, action,
        #                        reward, observation_, int(done))
        # agent.learn()
        # observation = observation_
        # n_steps += 1
    # scores.append(score)
    # steps_array.append(n_steps)

    # avg_score = np.mean(scores[-100:])
    # print('episode: ', i,'score: ', score,
    #       ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
    #       'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    # if avg_score > best_score:
    #     # if logg:
    #     # logger.save_models()
    #     # if not load_checkpoint:
    #     #    agent.save_models()
    #     best_score = avg_score

    # if logg:
    #     logger.save_score(score)

    # eps_history.append(agent.epsilon)
    # if load_checkpoint and n_steps >= 18000:
    #     break