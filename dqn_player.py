import numpy as np
import time
class DqnPlayer(object):
    def __init__(self,action_reward_board, is_study=1):
        self._is_study = is_study
        self.action_reward_board = action_reward_board
        self.EPSILON = 0.2
        self.FINAL_EPSILON = 0.0001
        self.EPSILON_REDUCE = (self.EPSILON - self.FINAL_EPSILON) / 1000000.0

    def set_is_study(self, is_study):
        self._is_study = is_study

    def softmax(self,x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    def get_action(self,normal_board, temp=0.8):
        if (len(normal_board.availables)) > 0:
            action_probs = np.zeros(normal_board.boardSize * (normal_board.boardSize - 1) * 2)
            rewards = self.action_reward_board(normal_board)
            rewards = rewards.reshape(60,)
            action_probs_softmax = self.softmax(1.0 / temp * rewards)

            action_probs[normal_board.availables] = action_probs_softmax[normal_board.availables]

            # 假如开场随意从中抽取一个action
            if len(normal_board.playRecord) == 0 or np.random.uniform() < self.EPSILON:
                final_act = np.random.choice(normal_board.availables)
                if self.FINAL_EPSILON < self.EPSILON: # random generate the action
                    self.EPSILON -= self.EPSILON_REDUCE
            else:
                final_act = action_probs.argmax(0) #by (1-epsilon) chance possibility, choose greedy action
            # use below code to see AI performance
            # final_act = action_probs.argmax(0)
            # time.sleep(2)
            return final_act
