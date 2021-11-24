class HyperParams:
    gamma = 0.999
    lamda = 0.97
    hidden = 64

    # UAV
    critic_lr_UAV = 0.0001
    actor_lr_UAV = 0.0001

    critic_lr_BS = 0.001
    actor_lr_BS = 0.001
    batch_size = 128
    l2_rate = 0.001
    max_kl = 0.05
    clip_param = 0.3

