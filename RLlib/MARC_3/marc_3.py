import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from .utils import soft_update, hard_update
from .agents import AccidentPolicy, FixationPolicy, QNetwork, StateEncoder, StateDecoder
from src.data_transform import scales_to_point, norm_fix
from metrics.losses import exp_loss, fixation_loss
import time


class MARC_3(object):
    def __init__(self, cfg, device=torch.device("cuda")):
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.beta_accident = cfg.beta_accident
        self.beta_fixation = cfg.beta_fixation
        self.losses = {}

        self.arch_type = cfg.arch_type
        self.type_acc = cfg.type_acc
        self.type_fix = cfg.type_fix
        self.actor_update_interval = cfg.actor_update_interval
        self.target_update_interval = cfg.target_update_interval
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.num_classes = cfg.num_classes
        self.device = device
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_shape
        self.input_size = cfg.input_shape
        self.pure_sl = cfg.pure_sl if hasattr(cfg, 'pure_sl') else False
        self.q_weight = cfg.q_weight
        self.regularization_weight = cfg.regularization_weight

        # state dims
        self.dim_state = cfg.dim_state
        self.dim_state_acc = int(0.5 * cfg.dim_state)
        self.dim_state_fix = int(0.5 * cfg.dim_state)
        # action dims
        self.dim_action_acc = cfg.dim_action_acc
        self.dim_action_fix = cfg.dim_action_fix
        self.dim_action = cfg.dim_action_acc + cfg.dim_action_fix

        # encoder
        self.encoder = StateEncoder(self.dim_state, cfg.dim_latent).to(device=self.device)
        # decoder
        self.decoder = StateDecoder(cfg.dim_latent, self.dim_state).to(device=self.device)

        # # create actor and critics
        self.policy_accident_1, self.policy_accident_2, self.policy_accident_3, self.policy_fixation_1, self.policy_fixation_2, self.policy_fixation_3, \
            self.critic_1, self.critic_2, self.critic_3, \
            self.critic_target_1, self.critic_target_2, self.critic_target_3, \
            self.policy_accident_target_1, self.policy_accident_target_2, self.policy_accident_target_3,\
             self.policy_fixation_target_1, self.policy_fixation_target_2, self.policy_fixation_target_3 \
                = self.create_actor_critics(cfg)
        hard_update(self.policy_accident_target_1, self.policy_accident_1)
        hard_update(self.policy_fixation_target_1, self.policy_fixation_1)
        hard_update(self.policy_accident_target_2, self.policy_accident_2)
        hard_update(self.policy_fixation_target_2, self.policy_fixation_2)
        hard_update(self.policy_accident_target_3, self.policy_accident_3)
        hard_update(self.policy_fixation_target_3, self.policy_fixation_3)
        hard_update(self.critic_target_1, self.critic_1)
        hard_update(self.critic_target_2, self.critic_2)
        hard_update(self.critic_target_3, self.critic_3)

        # optimizers
        self.critic_optim_1 = Adam(self.critic_1.parameters(), lr=cfg.critic_lr)
        self.critic_optim_2 = Adam(self.critic_2.parameters(), lr=cfg.critic_lr)
        self.critic_optim_3 = Adam(self.critic_3.parameters(), lr=cfg.critic_lr)
        self.policy_acc_optim_1 = Adam(self.policy_accident_1.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.policy_att_optim_1 = Adam(self.policy_fixation_1.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.policy_acc_optim_2 = Adam(self.policy_accident_2.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.policy_att_optim_2 = Adam(self.policy_fixation_2.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.policy_acc_optim_3 = Adam(self.policy_accident_3.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.policy_att_optim_3 = Adam(self.policy_fixation_3.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)

        if cfg.arch_type == 'rae':
            # the encoder is shared between two actors and critic, weights need to be tied
            self.policy_accident_1.state_encoder.copy_conv_weights_from(self.critic_1.state_encoder)
            self.policy_fixation_1.state_encoder.copy_conv_weights_from(self.critic_1.state_encoder)
            self.policy_accident_2.state_encoder.copy_conv_weights_from(self.critic_2.state_encoder)
            self.policy_fixation_2.state_encoder.copy_conv_weights_from(self.critic_2.state_encoder)
            self.policy_accident_3.state_encoder.copy_conv_weights_from(self.critic_3.state_encoder)
            self.policy_fixation_3.state_encoder.copy_conv_weights_from(self.critic_3.state_encoder)
            # decoder
            self.decoder = StateDecoder(cfg.dim_latent, self.dim_state).to(device=self.device)
            # optimizer for critic encoder 1 for reconstruction loss
            self.encoder_optim = Adam(self.critic_1.state_encoder.parameters(), lr=cfg.lr)
            # optimizer for decoder
            self.decoder_optim = Adam(self.decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.latent_lambda = cfg.latent_lambda
        
        if self.automatic_entropy_tuning:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.type_acc == "Gaussian" and not self.type_fix == 'Gaussian':
                dim_entropy = self.dim_action_acc
            elif not self.type_acc == "Gaussian" and self.type_fix == 'Gaussian':
                dim_entropy = self.dim_action_fix
            elif self.type_acc == "Gaussian" and self.type_fix == 'Gaussian':
                dim_entropy = self.dim_action
            else:
                print("When automatic entropy, at least one policy is Gaussian!")
                raise ValueError
            self.target_entropy = - dim_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
        


    def create_actor_critics(self, cfg):
        # create critic network 1
        critic_1 = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(device=self.device)
        critic_target_1 = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        # create critic network 2
        critic_2 = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(device=self.device)
        critic_target_2 = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        # create critic network 3
        critic_3 = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(device=self.device)
        critic_target_3 = QNetwork(self.dim_state, self.dim_action, cfg.hidden_size, dim_latent=cfg.dim_latent, arch_type=cfg.arch_type).to(self.device)
        # create accident anticipation policy
        dim_state = self.dim_state if cfg.arch_type == 'rae' else self.dim_state_acc
        policy_accident_1 = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        policy_accident_2 = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        policy_accident_3 = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        # create fixation prediction policy
        dim_state = self.dim_state if cfg.arch_type == 'rae' else self.dim_state_fix
        policy_fixation_1 = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)
        policy_fixation_2 = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)
        policy_fixation_3 = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)
        # create accident anticipation policy target
        dim_state = self.dim_state if cfg.arch_type == 'rae' else self.dim_state_acc
        policy_accident_target_1 = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        policy_accident_target_2 = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        policy_accident_target_3 = AccidentPolicy(dim_state, self.dim_action_acc, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_acc).to(self.device)
        # create fixation prediction policy target
        dim_state = self.dim_state if cfg.arch_type == 'rae' else self.dim_state_fix
        policy_fixation_target_1 = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)
        policy_fixation_target_2 = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)
        policy_fixation_target_3 = FixationPolicy(dim_state, self.dim_action_fix, cfg.hidden_size, 
            dim_latent=cfg.dim_latent, arch_type=cfg.arch_type, policy_type=self.type_fix).to(self.device)

        return policy_accident_1, policy_accident_2, policy_accident_3, \
        policy_fixation_1, policy_fixation_2, policy_fixation_3, \
        critic_1, critic_2, critic_3, critic_target_1, critic_target_2, critic_target_3, \
            policy_accident_target_1, policy_accident_target_2, policy_accident_target_3, \
            policy_fixation_target_1, policy_fixation_target_2, policy_fixation_target_3

    
    def set_status(self, phase='train'):
        isTraining = True if phase == 'train' else False
        self.policy_accident_1.train(isTraining)
        self.policy_fixation_1.train(isTraining)
        self.policy_accident_2.train(isTraining)
        self.policy_fixation_2.train(isTraining)
        self.policy_accident_3.train(isTraining)
        self.policy_fixation_3.train(isTraining)
        self.critic_1.train(isTraining)
        self.critic_target_1.train(isTraining)
        self.critic_2.train(isTraining)
        self.critic_target_2.train(isTraining)
        self.critic_3.train(isTraining)
        self.critic_target_3.train(isTraining)
        if self.arch_type == 'rae':
            self.encoder.train(isTraining) 
            self.decoder.train(isTraining)


    def select_action(self, state, rnn_state_1=None, rnn_state_2=None, rnn_state_3=None, evaluate=False):
        """state: (B, 64+64), [state_max, state_avg]
        """
        # initializing state
        state_max = state[:, :self.dim_state_acc]
        state_avg = state[:, self.dim_state_acc:]
        acc_state = state.clone() if self.arch_type == 'rae' else state_max
        fix_state = state.clone() if self.arch_type == 'rae' else state_avg

        # execute actions
        if evaluate is False:
            action_acc_1, rnn_state_1, _, _ = self.policy_accident_1.sample(acc_state, rnn_state_1)
            action_fix_1, _, _ = self.policy_fixation_1.sample(fix_state)
            action_acc_2, rnn_state_2, _, _ = self.policy_accident_2.sample(acc_state, rnn_state_2)
            action_fix_2, _, _ = self.policy_fixation_2.sample(fix_state)
            action_acc_3, rnn_state_3, _, _ = self.policy_accident_3.sample(acc_state, rnn_state_3)
            action_fix_3, _, _ = self.policy_fixation_3.sample(fix_state)
        else:
            _, rnn_state_1, _, action_acc_1 = self.policy_accident_1.sample(acc_state, rnn_state_1)
            _, _, action_fix_1 = self.policy_fixation_1.sample(fix_state)
            _, rnn_state_2, _, action_acc_2 = self.policy_accident_2.sample(acc_state, rnn_state_2)
            _, _, action_fix_2 = self.policy_fixation_2.sample(fix_state)
            _, rnn_state_3, _, action_acc_3 = self.policy_accident_3.sample(acc_state, rnn_state_3)
            _, _, action_fix_3 = self.policy_fixation_3.sample(fix_state)

        # get actions
        actions_1 = torch.cat([action_acc_1.detach(), action_fix_1.detach()], dim=1)  # (B, 3)
        actions_2 = torch.cat([action_acc_2.detach(), action_fix_2.detach()], dim=1)  # (B, 3)
        actions_3 = torch.cat([action_acc_3.detach(), action_fix_3.detach()], dim=1)  # (B, 3)

        q1 = self.critic_1(state, actions_1)
        q2 = self.critic_2(state, actions_2)
        q3 = self.critic_3(state, actions_3)

        actions = torch.ones(actions_1.shape)
        for i in range(q1.shape[0]):
            if q1[i] >= q2[i]:
                if q1[i] >= q3[i]:
                    action = actions_1[i]
                else:
                    action = actions_3[i]
            else:
                if q2[i] >= q3[i]:
                    action = actions_2[i]
                else:
                    action = actions_3[i]
            actions[i] = action
        actions = torch.Tensor(actions).to(self.device)

        if rnn_state_1 is not None:
            rnn_state_1 = (rnn_state_1[0].detach(), rnn_state_1[1].detach())
        if rnn_state_2 is not None:
            rnn_state_2 = (rnn_state_2[0].detach(), rnn_state_2[1].detach())
        if rnn_state_3 is not None:
            rnn_state_3 = (rnn_state_3[0].detach(), rnn_state_3[1].detach())

        return actions, rnn_state_1, rnn_state_2, rnn_state_3

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return target_net

    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, \
        rnn_state_batch_1, rnn_state_batch_2, rnn_state_batch_3, labels_batch, update_a=1):
        with torch.no_grad():
            # split the next_states
            next_state_max = next_state_batch[:, :self.dim_state_acc]
            next_state_avg = next_state_batch[:, self.dim_state_acc:]
            next_acc_state = next_state_batch.clone() if self.arch_type == 'rae' else next_state_max
            next_fix_state = next_state_batch.clone() if self.arch_type == 'rae' else next_state_avg

            # inference two policies
            next_acc_state_action_1, _, next_acc_state_log_pi, _ = self.policy_accident_target_1.sample(next_acc_state, rnn_state_batch_1)
            next_fix_state_action_1, next_fix_state_log_pi, _ = self.policy_fixation_target_1.sample(next_fix_state)
            next_acc_state_action_2, _, next_acc_state_log_pi, _ = self.policy_accident_target_2.sample(next_acc_state, rnn_state_batch_2)
            next_fix_state_action_2, next_fix_state_log_pi, _ = self.policy_fixation_target_2.sample(next_fix_state)
            next_acc_state_action_3, _, next_acc_state_log_pi, _ = self.policy_accident_target_3.sample(next_acc_state, rnn_state_batch_3)
            next_fix_state_action_3, next_fix_state_log_pi, _ = self.policy_fixation_target_3.sample(next_fix_state)

            next_state_action_1 = torch.cat([next_acc_state_action_1, next_fix_state_action_1], dim=1)
            next_state_action_2 = torch.cat([next_acc_state_action_2, next_fix_state_action_2], dim=1)
            next_state_action_3 = torch.cat([next_acc_state_action_3, next_fix_state_action_3], dim=1)

            # inference critics
                #### Compute the target Q values for action1
            qf1_next_target_a1 = self.critic_target_1(next_state_batch, next_state_action_1)
            qf2_next_target_a1 = self.critic_target_2(next_state_batch, next_state_action_1)
            qf3_next_target_a1 = self.critic_target_3(next_state_batch, next_state_action_1)
            
                #### Compute the target Q values for action2
            qf1_next_target_a2 = self.critic_target_1(next_state_batch, next_state_action_2)
            qf2_next_target_a2 = self.critic_target_2(next_state_batch, next_state_action_2)
            qf3_next_target_a2 = self.critic_target_3(next_state_batch, next_state_action_2)

                #### Compute the target Q values for action3
            qf1_next_target_a3 = self.critic_target_1(next_state_batch, next_state_action_3)
            qf2_next_target_a3 = self.critic_target_2(next_state_batch, next_state_action_3)
            qf3_next_target_a3 = self.critic_target_3(next_state_batch, next_state_action_3)

            ## min first, max afterward to avoid underestimation bias
            ## Taking minimum
            next_q1_aux = torch.min(qf1_next_target_a1, qf1_next_target_a2)
            next_Q1 = torch.min(next_q1_aux, qf1_next_target_a3)
            next_q2_aux = torch.min(qf2_next_target_a1, qf1_next_target_a2)
            next_Q2 = torch.min(next_q2_aux, qf2_next_target_a3)
            next_q3_aux = torch.min(qf3_next_target_a1, qf3_next_target_a2)
            next_Q3 = torch.min(next_q3_aux, qf3_next_target_a3)

            min_Q = torch.min(next_Q1, next_Q2)
            min_Q = torch.min(min_Q, next_Q3)

            max_Q = torch.max(next_Q1, next_Q2)
            max_Q = torch.max(min_Q, next_Q3)

            ## soft q update
            min_qf_next_target = self.q_weight * min_Q + (1-self.q_weight) * max_Q
                #### soft q update: next_Q = self.q_weight * torch.min(next_Q1, next_Q2) + (1-self.q_weight) * torch.max(next_Q1, next_Q2)

            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)
                #### target_Q = reward + not_done*self.discount*next_Q
            
        if update_a == 1:
            qf1 = self.critic_1(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf2 = self.critic_2(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf3 = self.critic_3(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

            # critic regularization
            qf1_loss = F.mse_loss(qf1, next_q_value) + self.regularization_weight * F.mse_loss(qf1, qf2) \
            + self.regularization_weight * F.mse_loss(qf2, qf3) + self.regularization_weight * F.mse_loss(qf3, qf1)

                #### Optimize
            self.critic_optim_1.zero_grad()
            qf1_loss.backward(retain_graph=True)
            self.critic_optim_1.step()
            self.losses.update({'critic': qf1_loss.item()})

            self.update_actor_1(state_batch, rnn_state_batch_1, labels_batch)

        elif update_a == 2:
            qf1 = self.critic_1(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf2 = self.critic_2(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf3 = self.critic_3(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

            # critic regularization
            qf2_loss = F.mse_loss(qf2, next_q_value) + self.regularization_weight * F.mse_loss(qf1, qf2) \
            + self.regularization_weight * F.mse_loss(qf2, qf3) + self.regularization_weight * F.mse_loss(qf3, qf1)

                #### Optimize
            self.critic_optim_2.zero_grad()
            qf2_loss.backward(retain_graph=True)
            self.critic_optim_2.step()
            self.losses.update({'critic_2': qf2_loss.item()})

            self.update_actor_2(state_batch, rnn_state_batch_2, labels_batch)
        else:
            qf1 = self.critic_1(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf2 = self.critic_2(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf3 = self.critic_3(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

            # critic regularization
            qf3_loss = F.mse_loss(qf3, next_q_value) + self.regularization_weight * F.mse_loss(qf1, qf2) \
            + self.regularization_weight * F.mse_loss(qf2, qf3) + self.regularization_weight * F.mse_loss(qf3, qf1)

                #### Optimize
            self.critic_optim_3.zero_grad()
            qf3_loss.backward(retain_graph=True)
            self.critic_optim_3.step()
            self.losses.update({'critic_3': qf3_loss.item()})

            self.update_actor_3(state_batch, rnn_state_batch_3, labels_batch)


    def update_actor_1(self, state_batch, rnn_state_batch_1, labels_batch):
        # split the states
        state_max = state_batch[:, :self.dim_state_acc]
        state_avg = state_batch[:, self.dim_state_acc:]
        acc_state = state_batch.clone() if self.arch_type == 'rae' else state_max
        fix_state = state_batch.clone() if self.arch_type == 'rae' else state_avg

        # sampling
        pi_acc, _, log_pi_acc, mean_acc = self.policy_accident_1.sample(acc_state, rnn_state_batch_1, detach=True)
        pi_fix, log_pi_fix, mean_fix = self.policy_fixation_1.sample(fix_state, detach=True)
        pi = torch.cat([pi_acc, pi_fix], dim=1)
        
        ## actor loss
        actor_loss = -self.critic_1(state_batch, pi).mean()

        # compute the early anticipation loss
        score_pred = 0.5 * (mean_acc + 1.0).squeeze(1)  # (B,)
        curtime_batch, clsID_batch, toa_batch, fix_batch = labels_batch[:, 0], labels_batch[:, 1], labels_batch[:, 2], labels_batch[:, 3:5]
        cls_target = torch.zeros(score_pred.size(0), self.num_classes).to(self.device)
        cls_target.scatter_(1, clsID_batch.unsqueeze(1).long(), 1)  # one-hot
        cls_loss = exp_loss(score_pred, cls_target, curtime_batch, toa_batch)  # exponential binary cross entropy

        # fixation loss
        mask = fix_batch[:, 0].bool().float() * fix_batch[:, 1].bool().float()  # (B,)
        fix_gt = fix_batch[mask.bool()]
        fix_pred = mean_fix[mask.bool()]
        fix_pred = scales_to_point(fix_pred, self.image_size, self.input_size)  # scaling scales to point
        fix_loss = torch.sum(torch.pow(norm_fix(fix_pred, self.input_size) - norm_fix(fix_gt, self.input_size), 2), dim=1).mean()  # (B) [0, sqrt(2)]

        if self.pure_sl:
            # for pure supervised learning, we just discard the losses from reinforcement learning
            acc_policy_loss = self.beta_accident * cls_loss
            fix_policy_loss = self.beta_fixation * fix_loss
        else:
            # weighted sum 
            acc_policy_loss = actor_loss.detach() + self.beta_accident * cls_loss
            fix_policy_loss = actor_loss.detach() + self.beta_fixation * fix_loss
        
        # update accident predictor
        self.policy_acc_optim_1.zero_grad()
        acc_policy_loss.backward()
        self.policy_acc_optim_1.step()
        # update attention predictor
        self.policy_att_optim_1.zero_grad()
        fix_policy_loss.backward()
        self.policy_att_optim_1.step()

        self.losses.update({'policy/total_accident': acc_policy_loss.item(),
                            'policy/actor': actor_loss.item(),
                            'policy/accident': cls_loss.item(),
                            'policy/total_fixation': fix_policy_loss.item(),
                            'policy/fixation': fix_loss.item()})
        soft_tau = 1e-2
        self.critic_target_1 = self.target_soft_update(self.critic_1, self.critic_target_1, soft_tau)
        self.policy_accident_target_1 = self.target_soft_update(self.policy_accident_1, self.policy_accident_target_1, soft_tau)
        self.policy_fixation_target_1 = self.target_soft_update(self.policy_fixation_1, self.policy_fixation_target_1, soft_tau)
     
    def update_actor_2(self, state_batch, rnn_state_batch_2, labels_batch):
        # split the states
        state_max = state_batch[:, :self.dim_state_acc]
        state_avg = state_batch[:, self.dim_state_acc:]
        acc_state = state_batch.clone() if self.arch_type == 'rae' else state_max
        fix_state = state_batch.clone() if self.arch_type == 'rae' else state_avg

        # sampling
        pi_acc, _, log_pi_acc, mean_acc = self.policy_accident_2.sample(acc_state, rnn_state_batch_2, detach=True)
        pi_fix, log_pi_fix, mean_fix = self.policy_fixation_2.sample(fix_state, detach=True)
        pi = torch.cat([pi_acc, pi_fix], dim=1)
        # log_pi = log_pi_acc + log_pi_fix

        # qf1_pi, qf2_pi = self.critic(state_batch, pi, detach=True)
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # actor loss
        # actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # actor_loss = - min_qf_pi.mean()

        # actor_loss = -self.critic.Q1(state_batch, pi, detach=True).mean()
        ## actor loss
        actor_loss = -self.critic_2(state_batch, pi).mean()

        # compute the early anticipation loss
        score_pred = 0.5 * (mean_acc + 1.0).squeeze(1)  # (B,)
        curtime_batch, clsID_batch, toa_batch, fix_batch = labels_batch[:, 0], labels_batch[:, 1], labels_batch[:, 2], labels_batch[:, 3:5]
        cls_target = torch.zeros(score_pred.size(0), self.num_classes).to(self.device)
        cls_target.scatter_(1, clsID_batch.unsqueeze(1).long(), 1)  # one-hot
        cls_loss = exp_loss(score_pred, cls_target, curtime_batch, toa_batch)  # exponential binary cross entropy

        # fixation loss
        mask = fix_batch[:, 0].bool().float() * fix_batch[:, 1].bool().float()  # (B,)
        fix_gt = fix_batch[mask.bool()]
        fix_pred = mean_fix[mask.bool()]
        fix_pred = scales_to_point(fix_pred, self.image_size, self.input_size)  # scaling scales to point
        fix_loss = torch.sum(torch.pow(norm_fix(fix_pred, self.input_size) - norm_fix(fix_gt, self.input_size), 2), dim=1).mean()  # (B) [0, sqrt(2)]

        if self.pure_sl:
            # for pure supervised learning, we just discard the losses from reinforcement learning
            acc_policy_loss = self.beta_accident * cls_loss
            fix_policy_loss = self.beta_fixation * fix_loss
        else:
            # weighted sum 
            acc_policy_loss = actor_loss.detach() + self.beta_accident * cls_loss
            fix_policy_loss = actor_loss.detach() + self.beta_fixation * fix_loss
        
        # update accident predictor
        self.policy_acc_optim_2.zero_grad()
        acc_policy_loss.backward()
        self.policy_acc_optim_2.step()
        # update attention predictor
        self.policy_att_optim_2.zero_grad()
        fix_policy_loss.backward()
        self.policy_att_optim_2.step()

        self.losses.update({'policy/total_accident': acc_policy_loss.item(),
                            'policy/actor': actor_loss.item(),
                            'policy/accident': cls_loss.item(),
                            'policy/total_fixation': fix_policy_loss.item(),
                            'policy/fixation': fix_loss.item()})
        soft_tau = 1e-2
        self.critic_target_2 = self.target_soft_update(self.critic_2, self.critic_target_2, soft_tau)
        self.policy_accident_target_2 = self.target_soft_update(self.policy_accident_2, self.policy_accident_target_2, soft_tau)
        self.policy_fixation_target_2 = self.target_soft_update(self.policy_fixation_2, self.policy_fixation_target_2, soft_tau)
    
    def update_actor_3(self, state_batch, rnn_state_batch_3, labels_batch):
        # split the states
        state_max = state_batch[:, :self.dim_state_acc]
        state_avg = state_batch[:, self.dim_state_acc:]
        acc_state = state_batch.clone() if self.arch_type == 'rae' else state_max
        fix_state = state_batch.clone() if self.arch_type == 'rae' else state_avg

        # sampling
        pi_acc, _, log_pi_acc, mean_acc = self.policy_accident_3.sample(acc_state, rnn_state_batch_3, detach=True)
        pi_fix, log_pi_fix, mean_fix = self.policy_fixation_3.sample(fix_state, detach=True)
        pi = torch.cat([pi_acc, pi_fix], dim=1)

        actor_loss = -self.critic_3(state_batch, pi).mean()

        # compute the early anticipation loss
        score_pred = 0.5 * (mean_acc + 1.0).squeeze(1)  # (B,)
        curtime_batch, clsID_batch, toa_batch, fix_batch = labels_batch[:, 0], labels_batch[:, 1], labels_batch[:, 2], labels_batch[:, 3:5]
        cls_target = torch.zeros(score_pred.size(0), self.num_classes).to(self.device)
        cls_target.scatter_(1, clsID_batch.unsqueeze(1).long(), 1)  # one-hot
        cls_loss = exp_loss(score_pred, cls_target, curtime_batch, toa_batch)  # exponential binary cross entropy

        # fixation loss
        mask = fix_batch[:, 0].bool().float() * fix_batch[:, 1].bool().float()  # (B,)
        fix_gt = fix_batch[mask.bool()]
        fix_pred = mean_fix[mask.bool()]
        fix_pred = scales_to_point(fix_pred, self.image_size, self.input_size)  # scaling scales to point
        fix_loss = torch.sum(torch.pow(norm_fix(fix_pred, self.input_size) - norm_fix(fix_gt, self.input_size), 2), dim=1).mean()  # (B) [0, sqrt(2)]

        if self.pure_sl:
            # for pure supervised learning, we just discard the losses from reinforcement learning
            acc_policy_loss = self.beta_accident * cls_loss
            fix_policy_loss = self.beta_fixation * fix_loss
        else:
            # weighted sum 
            acc_policy_loss = actor_loss.detach() + self.beta_accident * cls_loss
            fix_policy_loss = actor_loss.detach() + self.beta_fixation * fix_loss
        
        # update accident predictor
        self.policy_acc_optim_3.zero_grad()
        acc_policy_loss.backward()
        self.policy_acc_optim_3.step()
        # update attention predictor
        self.policy_att_optim_3.zero_grad()
        fix_policy_loss.backward()
        self.policy_att_optim_3.step()

        self.losses.update({'policy/total_accident': acc_policy_loss.item(),
                            'policy/actor': actor_loss.item(),
                            'policy/accident': cls_loss.item(),
                            'policy/total_fixation': fix_policy_loss.item(),
                            'policy/fixation': fix_loss.item()})
        soft_tau = 1e-2
        self.critic_target_3 = self.target_soft_update(self.critic_3, self.critic_target_3, soft_tau)
        self.policy_accident_target_3 = self.target_soft_update(self.policy_accident_3, self.policy_accident_target_3, soft_tau)
        self.policy_fixation_target_3 = self.target_soft_update(self.policy_fixation_3, self.policy_fixation_target_3, soft_tau)
                

    def update_decoder(self, state, latent_lambda=0.0):
        # encoder
        h = self.critic_1.state_encoder(state)
        # decoder
        state_rec = self.decoder(h)
        # MSE reconstruction loss
        rec_loss = F.mse_loss(state, state_rec) ## L_rec components # can try L1 loss and other kinds of losses
        # rec_loss = F.l1_loss(state, state_rec)

        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + latent_lambda * latent_loss
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()
        self.losses.update({'autoencoder': loss.item()})


    def update_parameters(self, memory, updates):
        
        # sampling from replay buffer memory
        state_batch, action_batch, reward_batch, next_state_batch, rnn_state_batch_1, rnn_state_batch_2, rnn_state_batch_3,\
             labels_batch, mask_batch = memory.sample(self.batch_size, self.device)

        if not self.pure_sl:
            # update critic networks
            self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, \
            rnn_state_batch_1, rnn_state_batch_2, rnn_state_batch_3, labels_batch, 1)
            self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, \
            rnn_state_batch_1, rnn_state_batch_2, rnn_state_batch_3, labels_batch, 2)
            self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, \
            rnn_state_batch_1, rnn_state_batch_2, rnn_state_batch_3, labels_batch, 0)
        
        # update decoder
        if self.arch_type == 'rae':
            self.update_decoder(state_batch, latent_lambda=self.latent_lambda)

        return self.losses


    def save_models(self, ckpt_dir, cfg, epoch):
        model_dict = {'policy_acc_model_1': self.policy_accident_1.state_dict(), 'policy_acc_model_2': self.policy_accident_2.state_dict(),
                      'policy_fix_model_1': self.policy_fixation_1.state_dict(), 'policy_fix_model_2': self.policy_fixation_2.state_dict(),
                      'policy_acc_model_3': self.policy_accident_3.state_dict(), 'policy_fix_model_3': self.policy_fixation_3.state_dict(),
                      'configs': cfg}
        torch.save(model_dict, os.path.join(ckpt_dir, 'sac_epoch_%02d.pt'%(epoch)))
        

    def load_models(self, ckpt_dir, cfg):
        if cfg.test_epoch == -1:
            filename = sorted(os.listdir(ckpt_dir))[-1]
            weight_file = os.path.join(cfg.output, 'checkpoints', filename)
        else:
            weight_file = os.path.join(cfg.output, 'checkpoints', 'sac_epoch_' + str(cfg.test_epoch).zfill(2) + '.pt')
            print(weight_file)
        if os.path.isfile(weight_file):
            checkpoint = torch.load(weight_file, map_location=self.device)
            self.policy_accident_1.load_state_dict(checkpoint['policy_acc_model_1'])
            self.policy_fixation_1.load_state_dict(checkpoint['policy_fix_model_1'])
            self.policy_accident_2.load_state_dict(checkpoint['policy_acc_model_2'])
            self.policy_fixation_2.load_state_dict(checkpoint['policy_fix_model_2'])
            self.policy_accident_3.load_state_dict(checkpoint['policy_acc_model_3'])
            self.policy_fixation_3.load_state_dict(checkpoint['policy_fix_model_3'])
            print("=> loaded checkpoint '{}'".format(weight_file))
        else:
            raise FileNotFoundError
