import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from dotsAndBoxesGUI import *
from dqn_player import DqnPlayer
from action_reward_net import *
from collections import deque
import random
import time
import pickle
np.set_printoptions(precision=6, suppress=True, edgeitems=20)
from PyQt5.QtCore import QThread,pyqtSignal
class ManagerTrain(QThread):
    actionOut = pyqtSignal(int)
    def __init__(self,gui_board,model_file=None,is_use_gpu=False):
        super(ManagerTrain, self).__init__()
        self.gui_board = gui_board
        self.normal_board = gui_board.to_normal_board()
        self.model_file = model_file
        self.is_use_gpu = is_use_gpu
        self.batch_size = 64
        self.lr = 1e-2
        self.lr_multiplier = 1.0
        self.action_reward_net_manager = ActionRewardNetManager(board_size=self.normal_board.boardSize,model_file=self.model_file,is_use_gpu=self.is_use_gpu)

        self.self_play_data_buffer = deque(maxlen=20000)
        # 增加了多少数据量 该变量增加随之增加
        # 每100000清零
        self.data_num = 0
        # 每100000加1
        self.data_base_num = 0

        self.epoch = 0

        self.dqn_player = None
        self.train_batch_epochs = 1
        self.evaluate_num = 10

        self.evaluate_fre = 10
        self.record_file = open("./record.txt", 'a+')


    # 自我对弈函数
    def selfplay_and_save_data(self,player, data_buffer):
        data, winner = self.normal_board.start_ai_play(player,actionOut=self.actionOut)
        data = list(data)[:]
        play_lens = (len(data))
        self.data_num += play_lens
        # if self.data_num > 100000:
        #     self.save_play_data()
        #     self.data_base_num += 1
        #     self.data_num = 0
        data_new = []
        # states   actions   next_states   reward   next_available  changes
        # 对数据进行维度置换 (4,5,5,4) ==> (4,4,5,5) =》data[i][0] data[i][1]
        for i in range(play_lens):
            data_state = np.transpose(data[i][0], (0, 3, 1, 2))
            next_state = np.transpose(data[i][2], (0, 3, 1, 2))
            data_new.append((data_state,data[i][1],next_state,data[i][3],data[i][4],data[i][5]))
        data_buffer.extend(data_new)
        return play_lens

    def policy_value_net_update(self,epoch):
        # data_buffer = copy.deepcopy(self.self_play_data_buffer)
        # random.shuffle(data_buffer)
        random.seed(None)
        batch = random.sample(self.self_play_data_buffer, self.batch_size)
        # states   actions   next_states   reward   next_available  changes
        states_batch = [data[0] for data in batch]
        actions_batch = [data[1] for data in batch]
        next_states_batch = [data[2] for data in batch]
        rewards_batch = [data[3] for data in batch]
        next_available_batch = [data[4] for data in batch]
        changes_batch = [data[5] for data in batch]

        # states_batch ==> (batch_size,4,4,5,5)
        # reward_old ==> (batch_size,60)
        # reward_old = self.action_reward_net_manager.policy_value(states_batch)
        # reward_old.flatten() ==> (batch_size*60,)
        # self.printf("reward_old: {}".format(reward_old.flatten()))
        for i in range(self.train_batch_epochs):
            loss = self.action_reward_net_manager.train_step(
                states_batch=states_batch,actions_batch=actions_batch,next_states_batch=next_states_batch,
                rewards_batch=rewards_batch,next_available_batch=next_available_batch,changes_batch=changes_batch,
                lr=self.lr*self.lr_multiplier,printf=self.printf
            )
        # reward_now = self.action_reward_net_manager.policy_value(states_batch)
        # self.printf("now_val：{}".format(reward_now.flatten()))
        self.printf("epoch:{}, Loss:{:.3f} ,self.lr{}, self.lr_multiplier{} ,lr{}".
                    format(epoch, loss ,self.lr,self.lr_multiplier,self.lr*self.lr_multiplier))

    def printf(self,str):
        print(str, file=self.record_file)
        print(str)
        self.record_file.flush()

    # save model
    def save_play_data(self):
        pickle.dump(self.self_play_data_buffer, open('play_data_DQN_{}_A'.format(self.data_base_num), 'wb+'))

    def save_model(self):
        self.action_reward_net_manager.save_model("./DQN_A_Now_{}.model".format(self.epoch))

    def open_play_data(self,data_file):
        self.self_play_data_buffer = pickle.load(open(data_file, 'rb'))

    def run(self):

        self.dqn_player = DqnPlayer(self.action_reward_net_manager.action_reward_board,is_study=1)
        self.printf(time.strftime("%Y-%m-%d %H:%M:%S"))
        self.printf("begin train!!!")

        # self.open_play_data("play_data_DQN_0_A")

        run_time = time.time()
        for i in range(1000000):
            i = i + 1000000
            self.epoch = i
            play_lens = self.selfplay_and_save_data(self.dqn_player,self.self_play_data_buffer)
            self.printf("i:{}, play_len:{},time:{}".format((i + 1), play_lens, time.time() - run_time))
            if len(self.self_play_data_buffer) > (self.batch_size*15):
                self.policy_value_net_update(epoch=i)
            # self.policy_value_net_update(epoch=i)
            if (i+1)%10000 == 0:
                self.action_reward_net_manager.save_model("./current_DQN_A_{}.model".format(i+1))
            run_time = time.time()
        self.record_file.close()



class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None,boardSize=6):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self,boardSize)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyMainWindow(boardSize=6)
    # "current_DQN_A_1000000.model"
    managerTrain = ManagerTrain(gui_board=ex.chessBoard,model_file="DQN_A_Now_2605.model",is_use_gpu=True)
    managerTrain.actionOut.connect(ex.chessBoard.do_action)
    ex.save_data.clicked.connect(managerTrain.save_play_data)
    ex.save_model.clicked.connect(managerTrain.save_model)
    managerTrain.start()
    ex.show()
    sys.exit(app.exec_())



