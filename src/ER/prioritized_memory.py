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

    def __init__(self, args, td_error, mask):
        self.alpha = args.selected_alpha
        self.res = torch.zeros_like(td_error)
        self.B, self.T, self.N =  td_error.shape
        self.mask = mask
        epsilon = mask * args.selected_epsilon
        td_error_epi = torch.abs(td_error) + epsilon
        td_error_epi_alpha = td_error_epi ** self.alpha
        self.prob = (td_error_epi_alpha / td_error_epi_alpha.sum())
        
        # beta
        self.max_step = args.t_max
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

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
    
    def sample_weight(self, n, step):
        sampled_pos = torch.multinomial(self.prob.reshape(-1), n, replacement=True) 
        N = self.B * self.T * self.N
        beta = (self.beta_end - self.beta_start)*step/self.max_step + self.beta_start
        weight = torch.pow(1 / (self.prob * N + 1e-8), beta) * self.mask 
        norm_weight = weight/ weight.max()

        index = sampled_pos
        pos_2 = index % self.N
        index = index // self.N
        pos_1 = index % self.T
        index = index // self.T
        pos_0 = index % self.B
        for i in range(n):
            self.res[pos_0[i],pos_1[i],pos_2[i]] += norm_weight[pos_0[i],pos_1[i],pos_2[i]]
        return self.res
            

