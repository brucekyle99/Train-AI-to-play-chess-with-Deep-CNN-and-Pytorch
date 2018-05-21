import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from dotsAndBoxesGUI import *
from alpha_player import AlphaPlayer
from mcts_player import MctsPlayer
from action_reward_net import *
from collections import deque
import random
import time
from PyQt5.QtCore import QThread,pyqtSignal
class ManagerTrain(QThread):
    actionOut = pyqtSignal(int)
    def __init__(self,gui_board,lr=15e-4,model_file=None,is_use_gpu=False):
        super(ManagerTrain, self).__init__()
        self.gui_board = gui_board
        self.normal_board = gui_board.to_normal_board()
        self.model_file = model_file
        self.is_use_gpu = is_use_gpu
        # set batch to 64、128、256、512
        self.batch_size = 256
        self.policy_value_net_manager_alpha = PolicyValueNetManager(board_size=self.normal_board.boardSize, lr=lr,
                                                              model_file=self.model_file, is_use_gpu=self.is_use_gpu)

        # 自我对弈的数据 以双向队列的形式进行存储 最大存储8000
        self.self_play_data_buffer = deque(maxlen=8000)

        # 敌手对弈的数据 以双向队列的形式进行存储 最大存储8000
        self.enemy_play_data_buffer = deque(maxlen=8000)

        self.alpha_player_one = None
        self.alpha_player_two = None
        # 对随机抽取出来的batch训练多少次
        self.train_batch_epochs = 40
        self.evaluate_num = 20
        self.alpha_player_simulation_num = 400
        self.mcts_player_simulation_num = 400


        # 每隔多少局实战评估一下alpha  即实战评估的频率
        self.evaluate_fre = 4

        # 当前Alpha最高的胜率：
        self.alpha_max_win_rate = 0
        self.alpha_constancy_win_num = 0

        # 打开一个文件用于存储相关信息  用于分析调试
        self.record_file = open("./TestModelRecord.txt", 'a+')

    def policy_evaluate(self, evaluate_num=10):
        current_mcts_player = MctsPlayer(c_puct=5,simulation_num=self.mcts_player_simulation_num)
        current_alpha_player = AlphaPlayer(self.policy_value_net_manager_alpha.policy_value_board, c_puct=5,
                                           simulation_num=self.alpha_player_simulation_num,is_study=0)

        # play_record[0]存储current_mcts_player的胜场
        # play_record[1]存储平局次数
        # play_record[2]存储current_mcts_player的败场
        play_record = [0,0,0]
        for i in range(evaluate_num):
            current_alpha_player.restart((-1 + (i % 2) * 2))
            current_mcts_player.restart(-(-1 + (i % 2) * 2))
            if current_alpha_player.get_player_id()==1:
                self.printf("Blue ALPHA")
            else:
                self.printf("Red ALPHA")
            winner = self.normal_board.strat_two_ai_play(current_mcts_player, current_alpha_player,actionOut=self.actionOut)
            # winner 输出为-1 0 1
            if winner != 0:
                if current_alpha_player.get_player_id() == winner:
                    play_record[0] += 1
                    self.printf("ALPHA Win")
                else:
                    play_record[2] += 1
                    self.printf("MTCS Win")
            else:
                play_record[1] += 1
        return play_record

    def printf(self,str):
        print(str, file=self.record_file)
        print(str)
        self.record_file.flush()

    def run(self):
        self.alpha_player_one = AlphaPlayer(self.policy_value_net_manager.policy_value_board,c_puct=5,
                                            simulation_num=self.alpha_player_simulation_num, is_study=1)
        self.printf(time.strftime("%Y-%m-%d %H:%M:%S"))
        self.printf("begin train!!!")
        run_time = time.time()
        # 1500为多少一次性 自我对弈/敌我对弈+训练
        for i in range(1500):
            evaluate_num = 10
            play_record = self.policy_evaluate(evaluate_num)
            alpha_win_rate = play_record[0] / evaluate_num
            self.printf("Alpha测试==> evaluate_num:{} ,mcts_player_simulation_num:{}, win: {}, lose: {}, tie:{}".format(
                evaluate_num,self.mcts_player_simulation_num, play_record[0], play_record[2], play_record[1]))
            self.policy_value_net_manager.save_model("./current_policy_value.model")
            # 如果当前的胜率大于之前最大的胜率
            if alpha_win_rate > self.alpha_max_win_rate:
                self.printf("The current policy exceeds the previous one !")
                self.alpha_max_win_rate = alpha_win_rate
                self.policy_value_net_manager.save_model("./better_policy_value.model")
                #假如alpha最大的胜率达到了1，则增加mcts模拟的次数
                if self.alpha_max_win_rate == 1.0:
                    self.mcts_player_simulation_num += 100
                    self.alpha_max_win_rate = 0.0
        self.record_file.close()



class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None,boardSize=3):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self,boardSize)

        # 可以在此处编写对应的signal和slot
        # 例如self.World.clicked.connect(self.onWorldClicked)
        # 下面写上对应的onWorldClicked方法函数




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyMainWindow(boardSize=6)
    managerTrain = ManagerTrain(gui_board=ex.chessBoard,model_file="current_policy_value_40.model", is_use_gpu=True)
    managerTrain.actionOut.connect(ex.chessBoard.do_action)
    managerTrain.start()
    ex.show()
    sys.exit(app.exec_())



