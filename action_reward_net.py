import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import torch.nn.init as init
import copy
# temporarily use CNN for training, can use ResNet、ResNeXt for better board feature extraction
class Net(nn.Module):
    def __init__(self, boardSize):
        super(Net, self).__init__()

        self.boardSize = boardSize

        self.conv1 = nn.Conv3d(4, 32, kernel_size=3, padding=1)  # padding=(kernel_size-1)/2
        init.xavier_uniform(self.conv1.weight.data,gain=init.calculate_gain("conv3d"))
        self.conv2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        init.xavier_uniform(self.conv2.weight.data, gain=init.calculate_gain("conv3d"))
        self.conv3 = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        init.xavier_uniform(self.conv3.weight.data, gain=init.calculate_gain("conv3d"))

        self.rew_fc1 = nn.Linear(8*100, 4*100)
        init.xavier_uniform(self.rew_fc1.weight.data, gain=init.calculate_gain("relu"))
        self.rew_fc2 = nn.Linear(4*100, 2*100)
        init.xavier_uniform(self.rew_fc2.weight.data, gain=init.calculate_gain("relu"))
        self.rew_fc3 = nn.Linear(2*100,60)


    def forward(self, data_input):
        # common layers
        # data_input维度为[5,5,5,4]
        x = F.relu(self.conv1(data_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv3(x))
        # reward
        x_act_reward = x.view(-1, 8*100) # flatten the conv output
        x_act_reward = F.relu(self.rew_fc1(x_act_reward))
        x_act_reward = F.relu(self.rew_fc2(x_act_reward))
        x_act_reward = self.rew_fc3(x_act_reward)

        return x_act_reward

class ActionRewardNetManager():
    def __init__(self, board_size, model_file=None, is_use_gpu=False):
        self.use_gpu = is_use_gpu
        self.board_size = board_size
        self.l2 = 1e-5
        self.gamma = 0.9
        self.learn_step_counter = 0
        self.target_replace_iter = 100
        #  use GPU to train or not
        if self.use_gpu:
            self.eval_action_reward_net, self.target_action_reward_net = Net(board_size).cuda(), Net(board_size).cuda()
        else:
            self.eval_action_reward_net, self.target_action_reward_net = Net(board_size), Net(board_size)
        self.optimizer = optim.Adam(self.eval_action_reward_net.parameters(),weight_decay=self.l2)

        # 假如model_file不为None，则加载模型文件
        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.eval_action_reward_net.load_state_dict(net_params)
            self.target_action_reward_net.load_state_dict(net_params)

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 数据格式输入检查 ok
    def action_reward_board(self, board):
        board_state_ = board.get_current_state()
        # board_state ( 4,5,5,4 ) => ( 4,4,5,5 )
        board_state = np.transpose(board_state_, (0, 3, 1, 2))
        if self.use_gpu:
            # 改为(*,4,4,5,5 )
            board_state = Variable(torch.FloatTensor([board_state])).cuda().float()
            actions_reward = self.target_action_reward_net(board_state)
            actions_reward = actions_reward.data.cpu().numpy()
        else:
            # 改为(*,4,4,5,5 )
            board_state = Variable(torch.FloatTensor([board_state])).float()
            actions_reward = self.target_action_reward_net(board_state)
            actions_reward = actions_reward.data.numpy()
        return actions_reward

    # 输入当前的states 6*6的 该方法用于测试debug
    def policy_value_state(self,board_state,availables):
        action_probs_list = []
        if self.use_gpu:
            ln_action_probs, value = self.policy_value_net(Variable(torch.FloatTensor(board_state)).cuda().float())
            action_probs = np.exp(ln_action_probs.data.cpu().numpy().flatten())
            for i in range(0, len(action_probs),60):
                action_probs_list.append(action_probs[i:i + 60])

        else:
            ln_action_probs, value = self.policy_value_net(Variable(torch.FloatTensor(board_state)).float())
            action_probs = np.exp(ln_action_probs.data.numpy().flatten())
            for i in range(0, len(action_probs),60):
                action_probs_list.append(action_probs[i:i + 60])

        if self.use_gpu:
            value = value.cpu().data.numpy()
        else:
            value = value.data.numpy()
        my_action_probs = []
        for i in range(len(availables)):
            my_action_probs.append(zip(availables[i], action_probs_list[i][availables[i]]))
        return my_action_probs, value

    # 输入当前的states (batch_size,4,4,5,5)
    def policy_value(self, board_state):
        # states (batch_size,4,4,5,5)
        if self.use_gpu:
            reward = self.target_action_reward_net(Variable(torch.FloatTensor(board_state)).cuda())
            reward = reward.cpu().data.numpy()
        else:
            reward = self.target_action_reward_net(Variable(torch.FloatTensor(board_state)))
            reward = reward.data.numpy()
        return reward


    def train_step(self, states_batch, actions_batch, next_states_batch, rewards_batch,next_available_batch, changes_batch, lr ,printf):
        BATCH_SIZE = len(states_batch)
        # states_batch_new = []
        # for i in range(BATCH_SIZE):
        #     states_batch_ = np.zeros((4, 10, 10))
        #     for k in range(4):
        #         states_batch_[k] = states_batch[i][k].reshape((10,10))
        #     states_batch_new.append(states_batch_)
        # states_batch = states_batch_new
        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
        if self.learn_step_counter % self.target_replace_iter == 0:
            printf("target_action_reward_net => load_state_dict")
            self.target_action_reward_net.load_state_dict(self.eval_action_reward_net.state_dict())
        self.learn_step_counter += 1

        # one_states_data = Variable(torch.FloatTensor([states_batch[0]]).cuda())
        if self.use_gpu:
            states_batch = Variable(torch.FloatTensor(states_batch).cuda())
            actions_batch = Variable(torch.LongTensor(actions_batch).cuda()).resize(BATCH_SIZE,1)
            next_states_batch = Variable(torch.FloatTensor(next_states_batch).cuda())
            rewards_batch = Variable(torch.FloatTensor(rewards_batch).cuda()).resize(BATCH_SIZE,1)
            next_available_batch = Variable(torch.FloatTensor(next_available_batch).cuda())
            changes_batch = Variable(torch.FloatTensor(changes_batch).cuda())
        else:
            states_batch = Variable(torch.FloatTensor(states_batch))
            actions_batch = Variable(torch.LongTensor(actions_batch)).resize(BATCH_SIZE,1)
            next_states_batch = Variable(torch.FloatTensor(next_states_batch))
            rewards_batch = Variable(torch.FloatTensor(rewards_batch)).resize(BATCH_SIZE,1)
            next_available_batch = Variable(torch.FloatTensor(next_available_batch))
            changes_batch = Variable(torch.FloatTensor(changes_batch))

        # 先将gradients置零
        # gradients set to 0
        self.optimizer.zero_grad()
        # 调整学习率, set learning rate
        self.adjust_learning_rate(self.optimizer, lr)

        # test_one_data = self.eval_action_reward_net(one_states_data)
        q_eval_all = self.eval_action_reward_net(states_batch)
        q_eval = q_eval_all.gather(1, actions_batch)
        q_target_next = self.target_action_reward_net(next_states_batch).detach() # detach(), no need to update
        # q_eval_next[0] 表示第一个样本所有动作的reward =》size 60
        q_eval_next = self.eval_action_reward_net(next_states_batch).detach()
        q_eval_next_exp_available = torch.exp(q_eval_next) * next_available_batch
        max_next_available_action = q_eval_next_exp_available.max(1)[1]
        selected_q_next = q_target_next[batch_index, max_next_available_action]

        q_target = rewards_batch + (self.gamma * selected_q_next * changes_batch).view(BATCH_SIZE, 1)

        loss = F.mse_loss(q_eval, q_target)
        loss.backward()
        self.optimizer.step()
        # q_eval_all = self.eval_action_reward_net(states_batch)
        # q_eval = q_eval_all.gather(1, actions_batch)
        return loss.data[0]

    # 获取当前神经网络的参数, get the NN parameters
    def get_net_param(self):
        params = self.target_action_reward_net.state_dict()
        return params

    # 将模型保存起来, save model
    def save_model(self, model_file_name):
        net_params = self.get_net_param() # get model params
        pickle.dump(net_params, open(model_file_name, 'wb'), protocol=2)