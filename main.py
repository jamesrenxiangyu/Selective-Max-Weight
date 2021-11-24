import datetime
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Actor, Critic
from utils import get_action
from utils import discrete_action
from utils import discrete_action_user
from collections import deque
from hparams import HyperParams as hp
import matplotlib.pyplot as plt
import math
import os
from ppo import train_model
import Env_UAV_cache
import warnings
warnings.filterwarnings("ignore", category=Warning)

if __name__ == "__main__":

    N_UAV = 4
    N_USER = 20

    N_TIME_SLOT = 200  # Number of time slots

    position_record = np.zeros((N_UAV, N_TIME_SLOT, N_TIME_SLOT))

    num_input = 69
    num_output = 8  # 动作A：V，theta
    num_output1 = N_USER # a

    actors = []
    critics = []
    actor_optims = []
    critic_optims = []
    for i in range(N_UAV):
        print(f"初始化agent_UAV_{i}")

        actor = Actor(num_input, num_output)
        critic = Critic(num_input)
        actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr_UAV)
        critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr_UAV, weight_decay=hp.l2_rate)

        actors.append(actor)
        critics.append(critic)
        actor_optims.append(actor_optim)
        critic_optims.append(critic_optim)
    
    actor_BS = Actor(num_input, num_output1)
    critic_BS = Critic(num_input)
    actor_BS_optim = optim.Adam(actor_BS.parameters(), lr=hp.actor_lr_BS)
    critic_BS_optim = optim.Adam(critic_BS.parameters(), lr=hp.critic_lr_BS, weight_decay=hp.l2_rate)
    print("初始化agent_BS")
    actors.append(actor_BS)
    critics.append(critic_BS)
    actor_optims.append(actor_BS_optim)
    critic_optims.append(critic_BS_optim)

    episodes = 0
    xar = []
    yar = []
    best_score = 0
    dic = {}

    iterations_num = 5
    for iter in range(iterations_num):

        memorys = []
        for i in range(N_UAV + 1):
            actors[i].eval(), critics[i].eval()
            memory = deque()
            memorys.append(memory)

        steps = 0
        scores = []

        for j in range(50):
            score = 0
            episodes += 1
            np.random.seed(1234)
            env = Env_UAV_cache.Environment()
            for i_step in range(1000000):
                action_all = np.zeros([N_UAV, 8])
                action_all_idx = 0
                s_all = []
                a_all = []
                for i in range(N_UAV):
                    state = env.observe_uav(i, i_step)
                    s_all.append(state)
                    mu, std, _ = actors[i](torch.Tensor(state).unsqueeze(0))
                    action = get_action(mu, std)[0]
                    action_discrete = discrete_action(action)

                    if env.UAV_pos[i][0] + action_discrete[0] * math.sin(math.radians( action_discrete[1]) * 90) > 1900 or env.UAV_pos[i][1] + action_discrete[0] * math.cos(math.radians(action_discrete[1]) * 90) > 1900 or env.UAV_pos[i][0] + action_discrete[0] * math.sin(math.radians( action_discrete[1]) * 90) <100 or env.UAV_pos[i][1] + action_discrete[0] * math.cos(math.radians(action_discrete[1]) * 90) < 100:
                        action[1] = (action[1] + 1) % 2
                        action_discrete[1] = (action_discrete[1] + 180) % 360

                    a_all.append(action)
                    action_all[action_all_idx, 0] = action_discrete[0]
                    action_all[action_all_idx, 1] = action_discrete[1]
                    action_all[action_all_idx, 2] = action_discrete[2]
                    action_all[action_all_idx, 3] = action_discrete[3]
                    action_all[action_all_idx, 4] = action_discrete[4]
                    action_all[action_all_idx, 5] = action_discrete[5]
                    action_all[action_all_idx, 6] = action_discrete[6]
                    action_all[action_all_idx, 7] = action_discrete[7]

                    action_all_idx += 1

                state = env.observe_uav(1,i_step)
                s_all.append(state)
                mu, std, _ = actors[N_UAV](torch.Tensor(state).unsqueeze(0))

                action_user = get_action(mu, std)[0]
                action_discrete_user = discrete_action_user(action_user)
                a_all.append(action_user)

                actions_user_temp = action_discrete_user.copy()
                actions_temp = action_all.copy()
                reward = env.step(actions_temp, actions_user_temp, i_step)

                done = 0
                mask = 1
                if i_step == N_TIME_SLOT-1:
                    done = 1
                    mask = 0

                for i in range(N_UAV + 1):
                    s = s_all[i]
                    a = a_all[i]
                    memorys[i].append([s, a, reward, mask])

                score += reward
                if done == 1:  # latency 退出
                    break

            scores.append(score)

        score_avg = np.mean(scores)
        cur_time = datetime.datetime.now()
        print(f"iter:{iter}, score_avg:{score_avg}, cur_time:{cur_time}")

        # if iter == iterations_num-1:
        #     best_score = score_avg
        #     checkpoint_dir = 'results'
        #     if not os.path.exists(checkpoint_dir):
        #         os.makedirs(checkpoint_dir)
        #     for j in range(N_UAV + 1):
        #         model_path = 'results/actor%d.pt' % j
        #         torch.save(actors[j].state_dict(), model_path)
        #         model_path = 'results/critic%d.pt' % j
        #         torch.save(critics[j].state_dict(), model_path)

        xar.append(int(episodes))
        yar.append(score_avg)
        for i in range(N_UAV + 1):
            actors[i].train(), critics[i].train()
            train_model(actors[i], critics[i], memorys[i], actor_optims[i], critic_optims[i])
        
    plt.plot(xar, yar)
    plt.show()
    np.savetxt('results/reward_iteration.txt', np.array(yar))



