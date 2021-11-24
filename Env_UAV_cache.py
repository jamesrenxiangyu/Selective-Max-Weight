from __future__ import division
import numpy as np
from numpy import random as nr
import math
import scipy.stats as stats

np.set_printoptions(suppress=True)
np.random.seed(10)

class Environment:
    def __init__(self):
        self.M = 4  # Number of UAVs
        self.U = 20  # Number of users

        ##### UAV Mobility Model
        self.height = 200  # UAV flight altitude

        self.MBS_pos = np.array([1000, 800, 0])
        self.UAV_pos = np.row_stack(([800, 600, self.height], [800, 1000, self.height], [1200, 600, self.height], [1200, 1000, self.height]))
        self.user_pos = np.row_stack(([1223, 805, 0], [1456, 1200, 0], [1050, 1350, 0], [1600, 920, 0], [1732, 1400, 0],
                                       [1100, 870, 0], [1032, 1500, 0], [432, 954, 0], [210, 1037, 0], [599, 1200, 0],
                                      [369, 1660, 0], [686, 996, 0], [777, 750, 0], [1150, 777, 0], [480, 330, 0],
                                       [1211, 232, 0], [1432, 685, 0], [1532, 1737, 0], [1632, 767, 0], [1699, 598, 0]))

        self.dt = 0.5  # Each time slot length
        self.N = 200  # Number of time slots

        ##### Cache Placement Model
        self.Cm = 5  # Storage capacity of each UAV
        self.Lf = 2  # Number of UAVs caching content f
        self.F = 30  # Content library
        self.user_request_content_library = np.zeros((self.N, self.U, self.F))

        ##### Transmission Channel Model
        self.dist_UAV_user = np.zeros((self.M, self.U))  # distance from UAV to user
        self.dist_MBS_UAV = np.zeros(self.M)
        self.dist_MBS_user = np.zeros(self.U)

        self.pathloss_UAV_user = np.zeros((self.M, self.U))
        self.pathloss_MBS_UAV = np.zeros(self.M)  # pathloss from MBS to UAV
        self.pathloss_MBS_user = np.zeros(self.U)

        self.interf_UAV_user = np.zeros((self.M, self.U))

        self.SINR_UAV_user = np.zeros((self.M, self.U))
        self.SINR_MBS_UAV = np.zeros(self.M)
        self.SINR_MBS_user = np.zeros(self.U)  # SINR from MBS to user

        self.f = 2 * pow(10, 9)  # Carrier frequency = 2GHz
        self.c = 3 * pow(10, 8)  # Speed of light = 3 * 10^8
        self.x_LoS = 1  # Shadowing factor 6dB, 20dB
        self.x_NLoS = 20
        self.c1 = 11.9  # Environment factor
        self.c2 = 0.13

        self.alpha = 2  # Path loss exponent
        self.eta = 100  # Additional path loss factor
        self.noise = pow(10, -10)  # Noise variance

        self.P_b = 2  # MBS transmit power

        self.F = 30  # Content library  index: 1~30
        self.F_range = np.arange(1, self.F)
        self.epilson = 1.1
        self.F_weights = self.F_range ** (-self.epilson)
        self.F_weights /= self.F_weights.sum()
        self.bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(self.F_range, self.F_weights))

        self.N = 200  # Number of time slots
        self.X = np.zeros((self.U, self.N))  # a content request matric
        for i in range(self.U):
            self.X[i] = self.bounded_zipf.rvs(size=200)

        self.W = 10000000  # Radio link bandwidth
        self.B = 20000000  # Backhual link bandwidth
        self.S = 10 * 1024 * 1024 # Size of each content

    def render(self,i_step):
        import matplotlib.pyplot as plt
        plt.scatter(self.UAV_pos[1, 0], self.UAV_pos[1, 1], c='#00CED1', alpha=0.4, label='UAV1')
        plt.scatter(self.UAV_pos[2, 0], self.UAV_pos[2, 1], c='#800080', alpha=0.4, label='UAV2')
        plt.scatter(self.UAV_pos[3, 0], self.UAV_pos[3, 1], c='#008B8B', alpha=0.4, label='UAV3')
        plt.scatter(self.UAV_pos[0, 0], self.UAV_pos[0, 1], c='#2E8B57', alpha=0.4, label='UAV4')

        plt.scatter(self.user_pos[:, 0], self.user_pos[:, 1], c='#DC143C', alpha=0.4, label='USER')
        plt.title("UAV" + str(i_step))
        plt.xlabel("X", fontsize=10)
        plt.ylabel("Y", fontsize=10)
        plt.xlim(0, 2000)
        plt.ylim(0, 2000)
        plt.tick_params(axis='both', labelsize=9)
        plt.show()

    def observe_uav(self, i, n_time_slot):
        bs_pos1 = self.UAV_pos.copy()
        user_pos1 = self.user_pos.copy()
        user_want1 = np.zeros(self.U+1)
        for i in range(self.U):
            user_want1[i] = self.X[i][n_time_slot]
        user_want1[self.U] = n_time_slot
        bs_pos1 = (bs_pos1 -1000) / 2000
        user_pos1 = (user_pos1 -800) / 1600
        return np.concatenate((np.reshape(bs_pos1[0:4][:,0:2], -1), np.reshape(user_want1, -1),np.reshape(user_pos1[:,0:2], -1)))

    def step(self, actions_all, User_association, n_time_slot):
        ## calc UAV position
        for i in range(self.M):
            self.UAV_pos[i][0] += actions_all[i][0] * math.sin(math.radians(actions_all[i][1]) * 90)
            self.UAV_pos[i][1] += actions_all[i][0] * math.cos(math.radians(actions_all[i][1]) * 90)

        ## calc UAV-to-user distance
        for i in range(self.M):
            for j in range(self.U):
                self.dist_UAV_user[i, j] = np.linalg.norm(self.UAV_pos[i] - self.user_pos[j])

        ## calc MBS-to-UAV distance
        for i in range(self.M):
            self.dist_MBS_UAV[i] = np.linalg.norm(self.MBS_pos - self.UAV_pos[i])

        ## calc MBS-to-user distance
        for i in range(self.U):
            self.dist_MBS_user[i] = np.linalg.norm(self.MBS_pos - self.user_pos[i])

        ## calc UAV-to-user pathloss
        for i in range(self.M):
            for j in range(self.U):
                theta = (180 / math.pi) * math.asin(self.height / self.dist_UAV_user[i, j])
                P_LoS = 1 / ((self.c1 * math.exp(-self.c2 * (theta - self.c1))) + 1)
                h_LoS = 20 * math.log((4 * math.pi * self.f * self.dist_UAV_user[i, j]/1000) / self.c) + self.x_LoS
                h_NLoS = 20 * math.log((4 * math.pi * self.f * self.dist_UAV_user[i, j]/1000) / self.c) + self.x_NLoS
                self.pathloss_UAV_user[i, j] = P_LoS * h_LoS + (1 - P_LoS) * h_NLoS

        ## calc MBS-to-UAV pathloss
    
        for i in range(self.M):
            theta = (180 / math.pi) * math.asin(self.height / self.dist_MBS_UAV[i])
            P_LoS = 1 / ((self.c1 * math.exp(-self.c2 * (theta - self.c1))) + 1)
            h_LoS = 20 * math.log((4 * math.pi * self.f * self.dist_MBS_UAV[i]/1000) / self.c) + self.x_LoS
            h_NLoS = 20 * math.log((4 * math.pi * self.f * self.dist_MBS_UAV[i]/1000) / self.c) + self.x_NLoS
            # h_LoS = math.pow(self.dist_MBS_UAV[i]/1000, -self.alpha)
            # h_NLoS = self.eta * h_LoS
            self.pathloss_MBS_UAV[i] = h_LoS + (1 - P_LoS) * h_NLoS

        ## calc MBS-to-user pathloss
        for i in range(self.U):
            self.pathloss_MBS_user[i] = 128.1 + 37.6 * math.log10(self.dist_MBS_user[i]/1000)

        ## calc UAV-to-user SINR
        self.interf_UAV_user = np.zeros((self.M, self.U))
        for i in range(self.M):
            for j in range(self.U):
                for k in range(self.M):
                    if k != i:
                        #print(k)
                        self.interf_UAV_user[i, j] += actions_all[k][2] * math.pow(10, -(self.pathloss_UAV_user[k, j]/10))
                self.SINR_UAV_user[i, j] = (actions_all[i][2] * math.pow(10, -(self.pathloss_UAV_user[i, j]/10)))/(self.interf_UAV_user[i, j] + self.noise)

        ## calc MBS-to-UAV SINR
        for i in range(self.M):
            self.SINR_MBS_UAV[i] = self.P_b / (self.noise * math.pow(10, (self.pathloss_MBS_UAV[i] / 10)))

        ## calc MBS-to-user SINR
        for i in range(self.U):
            self.SINR_MBS_user[i] = self.P_b / (self.noise * math.pow(10, (self.pathloss_MBS_user[i] / 10)))

        ## Cache Placement Model

        ## User Association Model
        # downlink transmission rate from access node i to user u at time slot n
        self.rate_MBS_USER = np.zeros(self.U)
        self.rate_UAV_USER = np.zeros((self.M, self.U))
        count_b = 0
        count_u1 = 0
        count_u2 = 0
        count_u3 = 0
        count_u4 = 0
        for i in range(self.U):
            if User_association[i] == 0:
                count_b += 1
            elif User_association[i] == 1:
                count_u1 += 1
            elif User_association[i] == 2:
                count_u2 += 1
            elif User_association[i] == 3:
                count_u3 += 1
            else:
                count_u4 += 1
        uav_user_1 = []
        uav_user_2 = []
        uav_user_3 = []
        uav_user_4 = []
        for i in range(self.U):
            if User_association[i] == 0:
                self.rate_MBS_USER[i] = self.W / count_b * math.log2(1+self.SINR_MBS_user[i])
            elif User_association[i] == 1:
                self.rate_UAV_USER[0][i] = self.W / count_u1 * math.log2(1+self.SINR_UAV_user[0][i])
                uav_user_1.append(i)
            elif User_association[i] == 2:
                self.rate_UAV_USER[1][i] = self.W / count_u2 * math.log2(1+self.SINR_UAV_user[1][i])
                uav_user_2.append(i)
            elif User_association[i] == 3:
                self.rate_UAV_USER[2][i] = self.W / count_u3 * math.log2(1+self.SINR_UAV_user[2][i])
                uav_user_3.append(i)
            else:
                self.rate_UAV_USER[3][i] = self.W / count_u4 * math.log2(1+self.SINR_UAV_user[3][i])
                uav_user_4.append(i)

        # backhual data transmission rate from MBS b to UAV m at time slot n
        UAV_cache = []
        UAV_cache.append([actions_all[0][3],actions_all[0][4],actions_all[0][5],actions_all[0][6],actions_all[0][7]])
        UAV_cache.append([actions_all[1][3],actions_all[1][4],actions_all[1][5],actions_all[1][6],actions_all[1][7]])
        UAV_cache.append([actions_all[2][3],actions_all[2][4],actions_all[2][5],actions_all[2][6],actions_all[2][7]])
        UAV_cache.append([actions_all[3][3],actions_all[3][4],actions_all[3][5],actions_all[3][6],actions_all[3][7]])

        uav_user_cache_1 = []
        uav_user_cache_2 = []
        uav_user_cache_3 = []
        uav_user_cache_4 = []
        for i in range(len(uav_user_1)):
            uav_user_cache_1.append(self.X[uav_user_1[i]][n_time_slot])
        for i in range(len(uav_user_2)):
            uav_user_cache_2.append(self.X[uav_user_2[i]][n_time_slot])
        for i in range(len(uav_user_3)):
            uav_user_cache_3.append(self.X[uav_user_3[i]][n_time_slot])
        for i in range(len(uav_user_4)):
            uav_user_cache_4.append(self.X[uav_user_4[i]][n_time_slot])

        self.T_MBS_UAV = np.zeros(self.M)

        Judge_y_u_f_1 = np.zeros(len(uav_user_1))
        Judge_y_u_f_2 = np.zeros(len(uav_user_2))
        Judge_y_u_f_3 = np.zeros(len(uav_user_3))
        Judge_y_u_f_4 = np.zeros(len(uav_user_4))
        Judge_1 = 1
        Judge_2 = 1
        Judge_3 = 1
        Judge_4 = 1
        for i in range(len(uav_user_cache_1)):
            if uav_user_cache_1[i] in UAV_cache[0]:
                Judge_y_u_f_1[i] = 1
            Judge_1 = Judge_1 * Judge_y_u_f_1[i]
        for i in range(len(uav_user_cache_2)):
            if uav_user_cache_2[i] in UAV_cache[1]:
                Judge_y_u_f_2[i] = 1
            Judge_2 = Judge_2 * Judge_y_u_f_2[i]
        for i in range(len(uav_user_cache_3)):
            if uav_user_cache_3[i] in UAV_cache[2]:
                Judge_y_u_f_3[i] = 1
            Judge_3 = Judge_3 * Judge_y_u_f_3[i]
        for i in range(len(uav_user_cache_4)):
            if uav_user_cache_4[i] in UAV_cache[3]:
                Judge_y_u_f_4[i] = 1
            Judge_4 = Judge_4 * Judge_y_u_f_4[i]

        Judge_all = (1-Judge_1) + (1-Judge_2) + (1-Judge_3) + (1-Judge_4)

        self.Delay = np.zeros(self.U)
        if Judge_all == 0:
            for i in range(self.U):
                if User_association[i] == 0:
                    self.Delay[i] = self.S / self.rate_MBS_USER[i]
                elif User_association[i] == 1:
                    self.Delay[i] = self.S / self.rate_UAV_USER[0][i]
                elif User_association[i] == 2:
                    self.Delay[i] = self.S / self.rate_UAV_USER[1][i]
                elif User_association[i] == 3:
                    self.Delay[i] = self.S / self.rate_UAV_USER[2][i]
                else:
                    self.Delay[i] = self.S / self.rate_UAV_USER[3][i]
        else:
            self.T_backhual = np.zeros(self.M)
            if Judge_1 == 0:
                self.T_backhual[0] = self.B / Judge_all * math.log2(1+self.SINR_MBS_UAV[0])
            if Judge_2 == 0:
                self.T_backhual[1] = self.B / Judge_all * math.log2(1+self.SINR_MBS_UAV[1])
            if Judge_3 == 0:
                self.T_backhual[2] = self.B / Judge_all * math.log2(1+self.SINR_MBS_UAV[2])
            if Judge_4 == 0:
                self.T_backhual[3] = self.B / Judge_all * math.log2(1+self.SINR_MBS_UAV[3])
            j1 = 0
            j2 = 0
            j3 = 0
            j4 = 0
            for i in range(self.U):
                if User_association[i] == 0:
                    self.Delay[i] = self.S / self.rate_MBS_USER[i]
                elif User_association[i] == 1:
                    if Judge_y_u_f_1[j1] == 1:
                        self.Delay[i] = self.S / self.rate_UAV_USER[0][i]
                    else:
                        self.Delay[i] = self.S / self.rate_UAV_USER[0][i] + self.S / self.T_backhual[0]
                    j1 += 1
                elif User_association[i] == 2:
                    if Judge_y_u_f_2[j2] == 1:
                        self.Delay[i] = self.S / self.rate_UAV_USER[1][i]
                    else:
                        self.Delay[i] = self.S / self.rate_UAV_USER[1][i] + self.S / self.T_backhual[1]
                    j2 += 1
                elif User_association[i] == 3:
                    if Judge_y_u_f_3[j3] == 1:
                        self.Delay[i] = self.S / self.rate_UAV_USER[2][i]
                    else:
                        self.Delay[i] = self.S / self.rate_UAV_USER[2][i] + self.S / self.T_backhual[2]
                    j3 += 1
                else:
                    if Judge_y_u_f_4[j4] == 1:
                        self.Delay[i] = self.S / self.rate_UAV_USER[3][i]
                    else:
                        self.Delay[i] = self.S / self.rate_UAV_USER[3][i] + self.S / self.T_backhual[3]
                    j4 += 1

        for i in range(self.U):
            self.Delay[i] = min(300, self.Delay[i])

        return 300 * 20 - sum(self.Delay)

# env = Environment()
# for i in range(1):    
#     a = env.step([[0,0,2,1,8,6,14,5],[0,0,2,4,6,1,21,9],[0,0,2,10,11,2,5,14],[0,0,2,11,10,15,18,19]],[1,1,1,1,1,4,4,1,2,2,2,2,2,1,1,1,3,3,3,4],i)
#     print(a)