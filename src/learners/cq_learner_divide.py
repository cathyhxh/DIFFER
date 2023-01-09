import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop, Adam
from ER.prioritized_memory import PER_Memory

class CQLearnerDivide:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.q_params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1: # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.mixer_params = list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.q_optimiser = RMSprop(params=self.q_params,
                                     lr=args.lr,
                                     alpha=args.optim_alpha,
                                     eps=args.optim_eps)
            self.mixer_optimiser = RMSprop(params=self.mixer_params,
                                     lr=args.lr,
                                     alpha=args.optim_alpha,
                                     eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.q_optimiser = Adam(params=self.q_params,
                                  lr=args.lr,
                                  eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.mixer_optimiser = Adam(params=self.mixer_params,
                                  lr=args.lr,
                                  eps=getattr(args, "optimizer_epsilon", 10E-8))
        
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        actions = batch["actions"][:, :-1]
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        indi_terminated = terminated.repeat(1,1,self.args.n_agents).float()
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):  # Note the minimum value of max_seq_length is 2
            agent_outs, _ = self.mac.forward(batch, actions=batch["actions"][:, t:t + 1].detach(), t=t)
            chosen_action_qvals.append(agent_outs)
        chosen_action_qvals = th.stack(chosen_action_qvals[:-1], dim=1)  # Concat over time

        best_target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            action_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True)
            best_target_actions.append(action_outs)
        best_target_actions = th.stack(best_target_actions, dim=1)  # Concat over time
        target_max_qvals = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t, actions=best_target_actions[:, t].detach())
            target_max_qvals.append(target_agent_outs)
        target_max_qvals = th.stack(target_max_qvals[1:], dim=1)  # Concat over time

        # Mix
        if self.mixer is not None:
            chosen_action_qvals_clone = chosen_action_qvals.view(-1, self.args.n_agents, 1).clone().detach()
            chosen_action_qvals_clone.requires_grad = True 
            target_max_qvals_clone = target_max_qvals.view(-1, self.args.n_agents, 1).clone().detach()

            chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])
            chosen_action_q_tot_vals = chosen_action_q_tot_vals.view(batch.batch_size, -1, 1)
            target_max_q_tot_vals = target_max_q_tot_vals.view(batch.batch_size, -1, 1)
        
        # Calculate 1-step Q-Learning targets
        targets = rewards.expand_as(target_max_q_tot_vals) + self.args.gamma * (1 -
                                                        terminated.expand_as(target_max_q_tot_vals)) * target_max_q_tot_vals
        
        # Td-error
        td_error = (chosen_action_q_tot_vals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        assert self.args.runner_scope == "episodic", "Runner scope HAS to be episodic if using rnn!"
        mixer_loss = (masked_td_error ** 2).sum() / mask.sum()


        # Optimise
        self.mixer_optimiser.zero_grad()
        chosen_action_qvals_clone.retain_grad() #the grad of qi
        chosen_action_q_tot_vals.retain_grad() #the grad of qtot
        mixer_loss.backward()

        grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8
        grad_l_qi = chosen_action_qvals_clone.grad.reshape(batch.batch_size, -1, self.args.n_agents)

        grad_qtot_qi = th.clamp(grad_l_qi/ grad_l_qtot, min=-10, max=10)#(B,T,n_agents)

        mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.mixer_params, self.args.grad_norm_clip)
        self.mixer_optimiser.step()

        chosen_action_qvals = chosen_action_qvals.view(batch.batch_size, -1, self.args.n_agents)
        target_max_qvals = target_max_qvals.view(batch.batch_size, -1, self.args.n_agents)
        q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals, indi_terminated) #(B,T,n_agents)
        q_rewards_clone = q_rewards.clone().detach()

        # Calculate 1-step Q-Learning targets
        q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals #(B,T,n_agents)

        # Td-error
        q_td_error = (chosen_action_qvals - q_targets.detach()) #(B,T,n_agents)

        q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents) #(B,T,n_agents)
        q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1, self.args.n_agents)
        # q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1])
        q_mask = q_mask.expand_as(q_td_error)

        masked_q_td_error = q_td_error * q_mask 
        q_selected_weight, selected_ratio = self.select_trajectory(masked_q_td_error.abs(), q_mask, t_env)
        q_selected_weight_clone = q_selected_weight.cuda().clone().detach()
        # 0-out the targets that came from padded data

        # Normal L2 loss, take mean over actual data
        q_loss = (masked_q_td_error ** 2 * q_selected_weight_clone).sum() / q_mask.sum()

        # Optimise
        self.q_optimiser.zero_grad()
        q_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()


        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = episode_num
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau = getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception("unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("q_loss", q_loss.item(), t_env)
            # self.logger.log_stat("q_grad_norm", q_grad_norm, t_env)
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / batch.batch_size), t_env)
            # self.logger.log_stat("q_taken_mean",
            #                      (chosen_action_qvals * mask).sum().item() / (batch.batch_size * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (batch.batch_size * self.args.n_agents),
            #                      t_env)
            self.logger.log_stat("selected_ratio", selected_ratio, t_env)
            self.logger.log_stat("mixer_loss", mixer_loss.item(), t_env)
            self.logger.log_stat("mixer_grad_norm", mixer_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("mixer_td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("mixer_target_mean", (targets * mask).sum().item()/mask_elems, t_env)

            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("q_grad_norm", q_grad_norm, t_env)
            q_mask_elems = q_mask.sum().item()
            self.logger.log_stat("q_td_error_abs", (masked_q_td_error.abs().sum().item()/q_mask_elems), t_env)
            self.logger.log_stat("q_q_taken_mean", (chosen_action_qvals * q_mask).sum().item()/(q_mask_elems), t_env)
            self.logger.log_stat("mixer_target_mean", (q_targets * q_mask).sum().item()/(q_mask_elems), t_env)
            self.logger.log_stat("reward_i_mean", (q_rewards * q_mask).sum().item()/(q_mask_elems), t_env)
            self.logger.log_stat("q_selected_weight_mean", (q_selected_weight_clone * q_mask).sum().item()/(q_mask_elems), t_env)

            self.log_stats_t = t_env


    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):
        # input: grad_qtot_qi (B,T,n_agents)  mixer_td_error (B,T,1)  qi (B,T,n_agents)  indi_terminated (B,T,n_agents)
        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents)) #(B,T,n_agents)
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i

    def select_trajectory(self, td_error, mask, t_env):
        # td_error (B, T, n_agents)
        if self.args.warm_up:
            if t_env/self.args.t_max<=self.args.warm_up_ratio:
                selected_ratio = t_env * (self.args.selected_ratio_end - self.args.selected_ratio_start)/(self.args.t_max * self.args.warm_up_ratio) + self.args.selected_ratio_start
            else:
                selected_ratio = self.args.selected_ratio_end
        else:
            selected_ratio = self.args.selected_ratio

        if self.args.selected == 'all':
            return th.ones_like(td_error).cuda(), selected_ratio
        elif self.args.selected == 'greedy':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error>=pivot, th.ones_like(td_error), th.zeros_like(td_error))
            return weight, selected_ratio
        elif self.args.selected == 'greedy_weight':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error>=pivot, td_error-pivot, th.zeros_like(td_error))
            norm_weight = weight/weight.max()
            return norm_weight, selected_ratio
        elif self.args.selected == 'PER':
            memory_size = int(mask.sum().item())
            memory = PER_Memory(memory_size)
            for b in range(mask.shape[0]):
                for t in range(mask.shape[1]):
                    for na in range(mask.shape[2]):
                        pos = (b,t,na)
                        if mask[pos] == 1:
                            memory.store(td_error[pos].cpu().detach(),pos)
            selected_num = int(memory_size * selected_ratio)
            mini_batch, selected_pos, is_weight = memory.sample(selected_num)
            weight = th.zeros_like(td_error)
            for idxs, pos in enumerate(selected_pos):
                weight[pos] += is_weight[idxs]
            return weight, selected_ratio
        elif self.args.selected == 'PER_hard':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return  PER_Memory(self.args, td_error, mask).sample(selected_num), selected_ratio
        elif self.args.selected == 'PER_weight':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return  PER_Memory(self.args, td_error, mask).sample_weight(selected_num, t_env), selected_ratio

    def _update_targets_soft(self, tau):

        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated target network (soft update tau={})".format(tau))

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))
        th.save(self.q_optimiser.state_dict(), "{}/q_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimiser.load_state_dict(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))
