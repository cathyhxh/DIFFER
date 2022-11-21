import numpy as np
np.random.seed(1)
import torch

class PER_Memory(object):  
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # epsilon = 0.01  # small amount to avoid zero priority
    # alpha = 0.6  # [0~1] convert the importance of TD error to priority
    # beta = 0.4  # importance-sampling, from initial value increasing to 1
    # beta_increment_per_sampling = 0.001
    # abs_err_upper = 1.  # clipped abs error

    def __init__(self, args, td_error):
        self.alpha = args.selected_alpha
        self.epsilon = args.selected_epsilon
        self.res = torch.zeros_like(td_error)
        self.B, self.T, self.N =  td_error.shape
        td_error_epi = torch.abs(td_error) + self.epsilon
        td_error_epi_alpha = td_error_epi ** self.alpha
        self.prob = (td_error_epi_alpha / td_error_epi_alpha.sum()).reshape(-1)

    def sample(self, n):
        sampled_pos = torch.multinomial(self.prob, n, replacement=True)        
        index = sampled_pos
        pos_2 = index % self.N
        index = index // self.N
        pos_1 = index % self.T
        index = index // self.T
        pos_0 = index % self.B
        for i in range(n):
            self.res[pos_0[i],pos_1[i],pos_2[i]] += 1

        return self.res
            

