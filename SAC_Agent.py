import numpy as np
import torch
import safety_gym
import gym
import time
# import numpy as np
import matplotlib.pyplot as plt
        
class ReplayBuffer:

    def __init__(self, s_size, capacity=50000):

        transition_type_str = '{0}float32, int, float32, {0}float32, float32, bool'.format(s_size)
        # transition_type_str = '{0}float32, int, float32, {0}float32, bool'.format(s_size)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=64)
        self.layer_2 = torch.nn.Linear(in_features=64, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output

class SACAgentWithCost:

    ALPHA_INITIAL = 1.
    REPLAY_BUFFER_BATCH_SIZE = 1000
    LEARNING_RATE = 5e-04
    # DISCOUNT_RATE = 0.7 #synthetic network
    # LEARNING_RATE = 0.001 #synthetic network
    LEARNING_RATE_ALPHA = 3e-06
    LEARNING_RATE_LAM = 0.03
    DISCOUNT_RATE = 0.95 #sf
    # LEARNING_RATE = 0.0001 #sf
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.01

    #Dual Update Parameters
    LAM_INITIAL = 10.0
    UPDATE_INTERVAL = 12

    def __init__(self, n_actions, s_size):
        self.state_dim = s_size
        self.action_dim = n_actions
        self.critic_local = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
        self.critic_local2 = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
        self.critic_target2 = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)


        self.actor_local = Network(input_dimension=self.state_dim, output_dimension=self.action_dim, output_activation=torch.nn.Softmax(dim=1) )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(s_size)

        self.target_entropy = 0.98 * -np.log(1 / n_actions)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        # self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE_ALPHA)
        self.state_value_mean=[]
        self.first_update = True
        
        self.critic_local_cost = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
        self.critic_local_cost2= Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
        self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target_cost = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
        self.critic_target_cost2 = Network(input_dimension=self.state_dim, output_dimension=self.action_dim)
 
        self.lam = torch.tensor(self.LAM_INITIAL, requires_grad=True)
        self.lam_optimiser = torch.optim.Adam([self.lam], lr=self.LEARNING_RATE_LAM)
        
        self.dual_interval = 0
        self.cost_lim = -5e-4
        # self.cost_lim = 10

        self.soft_update_target_networks(tau=1.)


    def act(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state[0,:])
        discrete_action = np.random.choice(range(self.action_dim), p=action_probabilities)
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state[0,:])
        discrete_action = np.argmax(action_probabilities)
        return discrete_action


    def train_model(self, state, discrete_action, reward, next_state, const, done):
        transition = (state[0,:], discrete_action, reward, next_state[0,:], const, done)
        self.train_networks(transition)
        
    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        self.critic_optimiser_cost.zero_grad()
        self.critic_optimiser_cost2.zero_grad()
        self.lam_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            self.dual_interval+=1
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]))
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]))
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            cost_tensor = torch.tensor(np.array(minibatch_separated[4]))
            done_tensor = torch.tensor(np.array(minibatch_separated[5]))

            critic_loss, critic_loss2, critic_loss_cost, critic_loss_cost2 = self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, cost_tensor, done_tensor)            
            # critic_loss, critic2_loss = self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)
            critic_loss.backward()
            self.gradient_clipping(self.critic_local.parameters(), 10.0)
            self.critic_optimiser.step()
            critic_loss2.backward()
            self.gradient_clipping(self.critic_local2.parameters(), 10.0)
            self.critic_optimiser2.step()

            critic_loss_cost.backward()
            self.gradient_clipping(self.critic_local_cost.parameters(), 10.0)
            self.critic_optimiser_cost.step()
            critic_loss_cost2.backward()
            self.gradient_clipping(self.critic_local_cost2.parameters(), 10.0)
            self.critic_optimiser_cost2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor, actions_tensor)
            # print(actor_loss)
            actor_loss.backward()
            self.gradient_clipping(self.actor_local.parameters(), 10.0)
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)
            alpha_loss.backward()
            # self.gradient_clipping(self.log_alpha.parameters(), 10.0)
            self.alpha_optimiser.step()
            # self.alpha = self.log_alpha.exp()

            if self.dual_interval==self.UPDATE_INTERVAL:
                self.dual_interval = 0
                lam_loss = self.lambda_loss(states_tensor, actions_tensor)
                lam_loss.backward()
                # self.gradient_clipping(self.lam.parameters(), 3.0)
                self.lam_optimiser.step()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, cost_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (torch.min(next_q_values_target, next_q_values_target2) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values
            
            if self.first_update:
                # self.state_value_mean.append(np.mean(next_q_values.numpy()))
                self.state_value_mean.append(np.mean(soft_state_values.numpy()))
                self.first_update = False

            next_q_values_target_cost = self.critic_target_cost.forward(next_states_tensor)
            next_q_values_target_cost2 = self.critic_target_cost2.forward(next_states_tensor)
            soft_state_values_cost = (action_probabilities * (torch.min(next_q_values_target_cost, next_q_values_target_cost2) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)
            # next_q_values_cost = (rewards_tensor+cost_tensor) + ~done_tensor * self.DISCOUNT_RATE*soft_state_values_cost
            next_q_values_cost = (cost_tensor) + ~done_tensor * self.DISCOUNT_RATE*soft_state_values_cost

                
        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        cse = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        cse_2 = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        critic_loss = cse.mean()
        critic_loss2 = cse_2.mean()

        soft_q_values_cost = self.critic_local_cost(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values_cost2 = self.critic_local_cost2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        cse_cost = torch.nn.MSELoss(reduction="none")(soft_q_values_cost, next_q_values_cost)
        cse_cost2 = torch.nn.MSELoss(reduction="none")(soft_q_values_cost2, next_q_values_cost)        
        critic_loss_cost = cse_cost.mean()
        critic_loss_cost2 = cse_cost2.mean()

        # weight_update = [min(min(l1.item(), l2.item()), min(c1.item(), c2.item())) for l1, l2, c1, c2 in zip(cse, cse_2, cse_cost, cse_cost2)]
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(cse, cse_2)]
        # weight_update = [min(c1.item(), c2.item()) for c1, c2 in zip(cse_cost, cse_cost2)]
        self.replay_buffer.update_weights(weight_update)

        return critic_loss, critic_loss2, critic_loss_cost, critic_loss_cost2

        # return critic_loss, critic2_loss

    def actor_loss(self, states_tensor, actions_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = (self.log_alpha.exp() * log_action_probabilities - torch.min(q_values_local, q_values_local2))#.mean(dim=1)

        q_values_cost = self.critic_local_cost(states_tensor)#.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        q_values_cost2 = self.critic_local_cost2(states_tensor)#.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        penalty = self.lam * torch.min(q_values_cost, q_values_cost2)
        lagrangian_loss = (action_probabilities * (inside_term+penalty)).sum(dim=1).mean()

        # print( policy_loss, penalty)

        # lagrangian_loss = policy_loss + penalty 
        return lagrangian_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha.exp() * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def lambda_loss(self, states_tensor, actions_tensor):
        q_cost = self.critic_local_cost(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        q_cost2 = self.critic_local_cost2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        # violation = torch.mean(torch.min(q_cost, q_cost2))  - self.cost_lim
        violation = torch.min(q_cost, q_cost2)  - self.cost_lim
        # violation = violation / violation.sum()
        
        # log_lam_tf=tf.nn.softplus(self.Lam.var)
        self.log_lam = torch.nn.functional.softplus(self.lam)
        lambda_loss =  self.log_lam*violation.detach()
        lambda_loss = -lambda_loss.sum(dim=-1)
        # print(self.lam)
        # violation_count = torch.where(q_cost  > self.cost_lim, torch.ones_like(q_cost ), torch.zeros_like(q_cost))
        # violation_rate = torch.reduce_sum(violation_count) / self.REPLAY_BUFFER_BATCH_SIZE        
        return lambda_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)
        self.soft_update(self.critic_target_cost, self.critic_local_cost, tau)
        self.soft_update(self.critic_target_cost2, self.critic_local_cost, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)

    def gradient_clipping(self, model_parameters, clip_val):
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=clip_val, norm_type=2)
        # torch.nn.utils.clip_grad_norm_(model_parameters, 1.0)


    def save(self, actor_path, critic_path):
        torch.save(self.actor_local.state_dict(), actor_path)        
        torch.save(self.critic_local.state_dict(), critic_path)        
        
    def load(self, actor_path, critic_path):
        self.actor_local.load_state_dict(torch.load(actor_path))        
        self.critic_local.load_state_dict(torch.load(critic_path))  
