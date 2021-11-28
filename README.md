# Selective-Max-Weight
Selective-Max-Weight (SMW) is an scheduling algorithm that considers the prority of each incoming packets
in terms of delay. SMW is also aimed at guaranteeing per-packet delay as well as achieving throughput optimal.

Since packet delay priority is introduced, the sequence of scheduling impacts the overall delay and throughput.
For delay sensitive packet, being late is as severe as being dropped. However, they may not expect packets arrive
too early. Therefore, there is a tradeoff in packet scheduling sequence: the 'arriving-early' packet can be 'delayed'
first, while the 'arriving late' packet can be scheduled earlier. 

This problem is NP-hard due to the arrival rate of each class of arriving packet is unknown, therefore, we introduce the
Reinforcement Learning technique to help us solve the maximization problem. 

The algorithm is still under developing...
