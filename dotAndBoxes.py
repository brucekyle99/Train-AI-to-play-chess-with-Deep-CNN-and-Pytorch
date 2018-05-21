import numpy as np
np.set_printoptions(precision=6, suppress=True)
from PyQt5.QtWidgets import QWidget,QLabel,QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import copy
import time
class NormalChessBoard(object):
    # boardSize这里应为4或者6  boardSize为4说明棋盘为4x4  boardSize为6说明棋盘为6x6
    def __init__(self, boardSize=4):
        self.boardSize = boardSize
        self.isEnd = False

        if boardSize == 4:
            self.firstBlockLoaction = 250
        elif boardSize == 6:
            self.firstBlockLoaction = 150

        self.chessNumMap = np.zeros(self.boardSize * (self.boardSize - 1) * 2)
        self.chessMap = np.zeros((self.boardSize - 1, self.boardSize - 1, 4))
        self.blockState = np.zeros((self.boardSize - 1, self.boardSize - 1))
        self.blockLocationList = []
        self.availables = list(range(self.boardSize * (self.boardSize - 1) * 2))
        self.chessNumToMap = []

        self.playRecord = []

        for i in range(boardSize - 1):
            for j in range(boardSize - 1):
                pix_x, pix_y = j * 100 + self.firstBlockLoaction, i * 100 + self.firstBlockLoaction
                self.blockLocationList.append((pix_x, pix_y))

        self.chessNumToBlockNum = []
        self.chessLocationPixList = []
        self.chessDirection = []

        self.rot_pros_mapped = []
        # 同上只是该变量用于左右翻转的情况
        self.fliplr_pros_mapped = []
        num = 0
        rotChessMap = np.zeros((self.boardSize - 1, self.boardSize - 1, 4))
        for i in range(boardSize - 1):
            for j in range(boardSize - 1):
                for k in range(4):
                    # 说明该边没有被访问过
                    if rotChessMap[i, j][k] == 0:
                        rotChessMap[i, j][k] = num
                        # k==0 说明边在块中心点的上面
                        if k == 0:
                            if (i - 1) >= 0:
                                rotChessMap[i - 1, j][2] = num
                        elif k == 1:
                            # 对已经标记过的位置赋值为1
                            if (j + 1) < (boardSize - 1):
                                rotChessMap[i, j + 1][3] = num
                        elif k == 2:
                            # 对已经标记过的位置赋值为1
                            if (i + 1) < (boardSize - 1):
                                rotChessMap[i + 1, j][0] = num
                        else:
                            # 对已经标记过的位置赋值为1
                            if (j - 1) >= 0:
                                rotChessMap[i, j - 1][1] = num
                        num += 1
        fliplrChessMap = copy.deepcopy(np.fliplr(rotChessMap))
        for i in range(boardSize - 1):
            for j in range(boardSize - 1):
                a = fliplrChessMap[i, j]
                fliplrChessMap[i, j] = np.roll(np.flip(a, 0), shift=1)

        rotChessMap = np.rot90(rotChessMap,k=-1)
        for i in range(boardSize - 1):
            for j in range(boardSize - 1):
                rotChessMap[i, j] = np.roll(rotChessMap[i, j], shift=1)

        # 初始化chessLocationPixList
        for i in range(boardSize - 1):
            for j in range(boardSize - 1):
                for k in range(4):
                    # 说明该边没有被访问过
                    if self.chessMap[i, j][k] == 0:
                        location = self.blockLocationList[i * (boardSize-1) + j]
                        # 将边对应的block编号存入List
                        self.chessNumToBlockNum.append((i * (boardSize-1)+ j, k))
                        # 将边对应的Map位置存入List
                        self.chessNumToMap.append((i, j, k))

                        self.rot_pros_mapped.append(rotChessMap[i,j,k])
                        self.fliplr_pros_mapped.append(fliplrChessMap[i,j,k])

                        # 对已经标记过的位置赋值为1
                        self.chessMap[i, j][k] = 1
                        # k==0 说明边在块中心点的上面
                        if k == 0:
                            location = (location[0], location[1] - 50)
                            self.chessDirection.append(0)
                            # 对已经标记过的位置赋值为1
                            if (i - 1) >= 0:
                                self.chessMap[i - 1, j][2] = 1
                        elif k == 1:
                            location = (location[0] + 50, location[1])
                            self.chessDirection.append(1)
                            # 对已经标记过的位置赋值为1
                            if (j + 1) < (boardSize - 1):
                                self.chessMap[i, j + 1][3] = 1
                        elif k == 2:
                            location = (location[0], location[1] + 50)
                            self.chessDirection.append(0)
                            # 对已经标记过的位置赋值为1
                            if (i + 1) < (boardSize - 1):
                                self.chessMap[i + 1, j][0] = 1
                        else:
                            location = (location[0] - 50, location[1])
                            self.chessDirection.append(1)
                            # 对已经标记过的位置赋值为1
                            if (j - 1) >= 0:
                                self.chessMap[i, j - 1][1] = 1
                        self.chessLocationPixList.append(location)

        self.chessMap = np.zeros((boardSize - 1, boardSize - 1, 4))

        '''
            chessMap存储棋盘状态 0表示为空  -1表示红棋  1表示蓝棋
        '''

        # 当前是否为蓝方下棋 1表示现在由蓝方进行下棋 -1表示红方下
        self.currentPlayer = 1
        self.step = 0
        self.isBegin = 0


    # 判断是否占领了块  这里直接遍历 效率稍低点 以后可以改进
    # 函数返回True表示占领了
    # 返回是否占领块 并对blockState信息进行更新
    def isGetBlock(self):

        # getBlockList里面存储的是块的编号
        getBlockList = []
        # 默认结束了，但是在后面的遍历中只要出现了一个空block就说明没有结束
        self.isEnd = True

        for i in range(self.boardSize - 1):
            for j in range(self.boardSize - 1):
                # 假如是0 说明之前没有被占则检查是否被占用
                if self.blockState[i, j] == 0:
                    sum = 0
                    for k in range(4):
                        if self.chessMap[i, j, k] != 0:
                            sum = sum + 1
                    # 假如sum等于4 说明该块由当前棋手（刚下完，还未换手时）占领了
                    if sum == 4:
                        # 更新blockState中该块信息
                        self.blockState[i, j] = self.currentPlayer
                        getBlockList.append(i * (self.boardSize - 1) + j)
                    else:
                        self.isEnd = False
        return getBlockList

    # 对棋盘状态无修改 判断下完后是否吃子
    def isHaveGetBlock(self):
        # getBlockList里面存储的是块的编号
        for i in range(self.boardSize - 1):
            for j in range(self.boardSize - 1):
                # 假如是0 说明之前没有被占则检查是否被占用
                if self.blockState[i, j] == 0:
                    sum = 0
                    for k in range(4):
                        if self.chessMap[i, j, k] != 0:
                            sum = sum + 1
                    if sum == 4:
                        return True
        return False

    def gameIsEnd(self):
        # 真正的判断在isGetBlock中，这里直接返回游戏是否结束信息,以及获胜者ID
        blueGetBlock = np.sum(self.blockState == 1)
        redGetBlock = np.sum(self.blockState == -1)
        if blueGetBlock > redGetBlock:
            winner = 1
        elif blueGetBlock < redGetBlock:
            winner = -1
        else:
            winner = 0
        # 辅助判断 假如一方占有的块超过了总块数的一半则直接判赢
        if blueGetBlock > ((self.blockState.size) // 2) or redGetBlock > ((self.blockState.size) // 2):
            self.isEnd = True

        return self.isEnd, winner, blueGetBlock, redGetBlock

    def reSetChess(self):
        # print("重新开始！")
        self.chessMap = np.zeros((self.boardSize - 1, self.boardSize - 1, 4))
        self.blockState = np.zeros((self.boardSize - 1, self.boardSize - 1))
        self.playRecord = []
        self.chessNumMap = np.zeros(self.boardSize * (self.boardSize - 1) * 2)
        self.isEnd = False
        # 1表示蓝方下  0表示红方下
        self.currentPlayer = 1
        self.step = 0
        self.availables = list(range(self.boardSize * (self.boardSize - 1) * 2))


    def setChess(self, chessMap_):
        self.chessMap = chessMap_

    def isBuleDown(self):
        if self.currentPlayer == 1:
            return True
        else:
            return False

    # 交替换手 其中包括检查游戏是否结束
    # showResult默然为0 表示弹出框来显示最后结果
    def alternate(self):
        # getBlockList存储的是被当前下棋者占有的块的编号
        getBlockList = self.isGetBlock()

        # 假如list为空
        if not getBlockList:
            self.currentPlayer = -self.currentPlayer


    def startPlay(self):
        self.isBegin = 1
        self.reSetChess()

    # 用于判断当前位置是否属于棋盘上棋子有效点击范围内
    # 输入一个位置和一个棋编号
    # 返回True表示在该棋子的点击范围
    def isChessDomain(self, location, chessNum):
        x, y = location[0], location[1]
        x0, y0 = 0, 0
        if chessNum >= 0 and chessNum < self.chessNumMap.size:
            x0, y0 = self.chessLocationPixList[chessNum][0], self.chessLocationPixList[chessNum][1]
        else:
            print("isChessDomain-------chessNum异常!!  当前chessNum：", chessNum)
        return (((y - y0) > (abs(x - x0) - 50)) and ((y - y0) < (-abs(x - x0) + 50)))

    # 像素转边的棋编号  该方法效率低 有待改进
    # 输入(x,y)
    def pixelToChessNum(self, location):
        for i in range(self.boardSize * (self.boardSize - 1) * 2):
            # 假如location在编号i的棋的范围内 返回该编号
            if self.isChessDomain(location, i):
                return i
        return -1

    # 输入棋编号 改变对应的chessMap
    def changeChessStatus(self, chessNum):
        blockNum, k = self.chessNumToBlockNum[chessNum]
        # 改变棋盘状态信息chessMap
        i, j = blockNum // (self.boardSize - 1), blockNum % (self.boardSize - 1)
        self.chessMap[i, j, k] = self.currentPlayer

        # 假如k为 3或者0的话是不用更新的
        # 因为假如k为3说明是最左侧的边 假如是0的话说明是最上面的边

        # 假如是横向的话同步对下方的block进行更新
        # 假如是竖向的话同步对右侧的block进行更新

        # 当k值为1说明是block的第二条边 为竖直方向
        if k == 1:
            if (j + 1) < (self.boardSize - 1):
                self.chessMap[i, j + 1, 3] = self.currentPlayer
        if k == 2:
            if (i + 1) < (self.boardSize - 1):
                self.chessMap[i + 1, j, 0] = self.currentPlayer
        # 改变棋盘状态信息chessNumMap、availables
        self.chessNumMap[chessNum] = self.currentPlayer
        self.availables.remove(chessNum)
        # 将下棋人及其所下的位置进行记录
        self.playRecord.append((self.currentPlayer, chessNum))

    def regainChessStatus(self,chessNum):
        blockNum, k = self.chessNumToBlockNum[chessNum]
        # 改变棋盘状态信息chessMap
        i, j = blockNum // (self.boardSize - 1), blockNum % (self.boardSize - 1)
        self.chessMap[i, j, k] = 0

        if k == 1:
            if (j + 1) < (self.boardSize - 1):
                self.chessMap[i, j + 1, 3] = 0
        if k == 2:
            if (i + 1) < (self.boardSize - 1):
                self.chessMap[i + 1, j, 0] = 0
        # 改变棋盘状态信息chessNumMap、availables
        self.chessNumMap[chessNum] = 0
        # todo 要求恢复之前的 availables
        self.availables.append(chessNum)
        self.availables = sorted(self.availables)
        # 将下棋人及其所下的位置进行记录
        self.playRecord.remove(self.playRecord[-1])

    # chessNum为棋盘中标号 type为绘制棋子的类型
    # 0表示正常
    # 1表示含有红色圈圈的表示最后一次下的棋子
    # 2表示半透明的棋子
    def drawByChessNum(self, chessNum):
        # 检查chessNum
        if chessNum < 0 and chessNum >= self.chessNumMap.size:
            print("drawByChessNum-------chessNum异常!!  当前chessNum：", chessNum)
            return
        # # 当x,y处于棋盘的范围之内，且该位置上没有棋子时，对其进行绘制
        if self.chessNumMap[chessNum] == 0:
            pix_x, pix_y = self.chessLocationPixList[chessNum]
            # 改变棋盘状态信息chessMap
            self.changeChessStatus(chessNum)

            # 将上一步的棋子样式修改为normal状态
            if self.step >= 1:
                player, chessNum_ = self.playRecord[self.step - 1]

            self.step += 1

    # action即chessNum
    # showResult即是否显示对话框
    # autoJudge是否自动判断比赛输赢（即是否在我方下完棋，当前棋手为对手或者我方时自动进行判断输赢情况）
    # autoReset是否在判断游戏结束后自动重新开始？注意假如autoJudge=0则autoReset毫无意义

    def do_action(self, action, showDialog=1, autoJudge=1, autoReset=1):
        self.drawByChessNum(action)
        self.alternate()
        # 下面是判断输赢 autoJudge为1表示自动判断
        if autoJudge != 1:
            return
        isEnd, winner, blueGetBlock, redGetBlock = self.gameIsEnd()
        if isEnd:
            # print(self.blockState)
            # print(self.chessNumMap)
            # print(self.chessMap)

            if blueGetBlock > redGetBlock:
                winner = 1
                # print('Nor 蓝方胜出！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
            elif redGetBlock > blueGetBlock:
                winner = -1
                # print('Nor 红方胜出！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
            else:
                winner = 0
                # print("Nor 平局")
            if autoReset:
                self.reSetChess()

    # action即chessNum
    # 该方法提供给AI进行调用
    # 执行完对应的action之后返回当前棋局对应的currentPlayer
    def ai_do_action(self, action):
        self.do_action(action, 0, 0, 0)
        return self.currentPlayer

    # 改为4层数据
    def get_current_state(self):
        data_state = np.zeros((4, (self.boardSize - 1), (self.boardSize - 1), 4))
        # 第一层：当前棋局chessMap ((boardSize-1)*(boardSize-1)*4)
        data_state[0] = copy.deepcopy(self.chessMap)
        for i in range(self.boardSize - 1):
            for j in range(self.boardSize - 1):
                for k in range(4):
                    if data_state[0][i, j, k] == -1:
                        data_state[0][i, j, k] = 1

        # 第二层 第三层：我方所占的位置 对方所占位置
        for i in range(self.boardSize - 1):
            for j in range(self.boardSize - 1):
                for k in range(4):
                    if self.blockState[i, j] == self.currentPlayer:
                        data_state[1][i, j, k] = abs(self.blockState[i, j])
                    else:
                        data_state[2][i, j, k] = abs(self.blockState[i, j])


        # 第三层：上一步棋子位置   ((boardSize-1)*(boardSize-1)*4) 棋子的位置为1其他位置为0
        # if self.playRecord:
        #     player, chessNum = self.playRecord[-1]
        #     mapIndex = self.chessNumToMap[chessNum]
        #     data_state[3,mapIndex[0],mapIndex[1],mapIndex[2]] = 1
        #     # mapIndex[2]为0不用管，mapIndex[2]为3也不用管
        #     if mapIndex[2]==1 and (mapIndex[1]+1)<(self.boardSize-1):
        #         data_state[3, mapIndex[0], mapIndex[1]+1, 3] = 1
        #     if mapIndex[2]==2 and (mapIndex[0]+1)<(self.boardSize-1):
        #         data_state[3, mapIndex[0]+1, mapIndex[1], 0] = 1

        # 第四层：当前棋手是否先手   先手为1  后手为0   ((boardSize-1)*(boardSize-1)*4)
        # 这里我们默认是蓝方先手
        if self.isBuleDown():
            data_state[3] = np.ones((self.boardSize - 1,self.boardSize - 1,4))
        return data_state

    # 只下一盘棋
    def start_ai_play(self,player, actionOut = None):
        # 初始化比赛
        self.reSetChess()
        # states   actions   next_states   reward   next_available  changes
        states, actions, next_states, rewards, next_available, changes = [], [], [] ,[], [], []
        # 默认蓝方先下
        self.currentPlayer = 1
        # 默认蓝方先下
        t1 = time.time()
        sumBlock = 0
        while (1):
            action = player.get_action(self)
            # time.sleep(1)
            # 添加当前的state状态、及采取的动作action
            states.append(self.get_current_state())
            actions.append(int(action))
            # print("start_ai_play 调试输出：currentPlayer：{} action：{} time：{}".format(self.currentPlayer, action,
            #                                                                      time.time() - t1))
            t1 = time.time()
            if actionOut is not None:
                actionOut.emit(action)
            self.ai_do_action(action)
            isEnd, winner, blueGetBlock, redGetBlock = self.gameIsEnd()
            reward = (blueGetBlock+redGetBlock)-sumBlock
            sumBlock = blueGetBlock + redGetBlock
            # 添加采取动作action之后的状态、及可执行动作
            next_states.append(self.get_current_state())
            next_available.append((copy.deepcopy(self.chessNumMap)+1)%2)
            # 假如有回报说明换手
            if reward != 0:
                changes.append(1.0)
            else:
                # 换手
                changes.append(-1.0)
            # print("isEnd：{}, winner：{}, blueGetBlock：{}, redGetBlock：{}".format(
            #     isEnd, winner, blueGetBlock, redGetBlock
            # ))
            # 下完判断是否结束
            if isEnd:
                # 添加采取动作action之后的立即回报
                rewards.append(0.5)
                print("start_ai_play == 结束")
                # 假如结束了，判断player_one赢还是player_two赢，还是平局
                # 假设是平局
                if winner == 0:
                    print('start_ai_play 平局！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
                # 假设非平局
                else:
                    if winner == 1:
                        print('start_ai_play 蓝方胜出！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
                    else:
                        print('start_ai_play 红方胜出！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
                # states   actions   next_states   reward   next_available  changes
                data = zip(states, actions, next_states, rewards, next_available, changes)
                data = list(data)[:]
                # data 为list [(states[0],probs[0],board_values),(states[1],probs[1],board_values),()]
                # states[0]==>(6,3,3,4)  probs[0]==>(24,)
                # data_ = [data[5]]
                # todo 使用extend_data对当前的数据进行增量处理 该方法需要改写
                # data = self.extend_data(data)
                return data, winner
            else:
                # 添加采取动作action之后的立即回报
                if reward != 0:
                    rewards.append(0.3)
                else:
                    rewards.append(0.0)


    def strat_two_ai_play(self,player_one,player_two,actionOut = None):
        # 初始化比赛
        self.reSetChess()
        # 默认蓝方先下
        self.currentPlayer = 1
        t1 = time.time()
        while(1):
            if self.currentPlayer == player_one.get_player_id():
                player = player_one
            else:
                player = player_two
            action, action_probs = player.get_action(self)
            print("start_ai_play 调试输出：currentPlayer：{} action：{} time：{}".
                  format(self.currentPlayer, action,time.time() - t1))
            t1 = time.time()
            if actionOut is not None:
                actionOut.emit(action)
            self.ai_do_action(action)
            isEnd, winner, blueGetBlock, redGetBlock = self.gameIsEnd()
            if isEnd:
                print("isEnd：{}, winner：{}, blueGetBlock：{}, redGetBlock：{}".format(
                    isEnd, winner, blueGetBlock, redGetBlock
                ))
            # 下完判断是否结束
            if isEnd:
                print("start_ai_play == 结束")
                # 假如是测试则直接返回比赛结果
                return winner

    # todo 扩充样本的函数
    def extend_data(self, data):
        extend_data = []
        for state_, prob_, board_value_ in data:
            state = copy.deepcopy(state_)
            prob = copy.deepcopy(prob_)
            extend_data += self.trans_data(state,prob,board_value_)

        return extend_data


    # 给出一个状态 返回8个状态
    def trans_data(self,state_,prob_,board_value_):
        extend_data = []
        extend_data.append((state_, prob_, board_value_))
        state = copy.deepcopy(state_)
        # 对其进行左右翻转 得到state_fliplr、prob_fliplr
        state_fliplr = self.trans_state(state,mode=1)
        prob_fliplr = np.zeros(prob_.shape)
        for i in range(prob_.size):
            prob_fliplr[i] = prob_[int(self.fliplr_pros_mapped[i])]
        extend_data.append((state_fliplr,prob_fliplr,board_value_))

        # 使用原数据进行3次90度旋转
        prob_temp = copy.deepcopy(prob_)
        state_temp = copy.deepcopy(state_)
        for i in range(3):
            state_rot = self.trans_state(state_temp)
            prob_rot = np.zeros(prob_.shape)
            for i in range(prob_.size):
                prob_rot[i] = prob_temp[int(self.rot_pros_mapped[i])]
            extend_data.append((state_rot,prob_rot,board_value_))
            state_temp = state_rot
            prob_temp = prob_rot

        # 使用翻转数据进行3次90度旋转
        prob_temp = copy.deepcopy(prob_fliplr)
        state_temp = copy.deepcopy(state_fliplr)
        for i in range(3):
            state_rot = self.trans_state(state_temp)
            prob_rot = np.zeros(prob_.shape)
            for i in range(prob_.size):
                prob_rot[i] = prob_temp[int(self.rot_pros_mapped[i])]
            extend_data.append((state_rot, prob_rot, board_value_))
            state_temp = state_rot
            prob_temp = prob_rot

        return extend_data

    def trans_state(self,state_,mode=0):
        state = copy.deepcopy(state_)
        # 假如顺时针翻转
        if mode == 0:
            for k in range(3):
                state[k] = copy.deepcopy(np.rot90(state_[k], k=-1))
                for i in range(self.boardSize - 1):
                    for j in range(self.boardSize - 1):
                        state[k, i, j] = np.roll(state[k, i, j], shift=1)
            return state
        if mode == 1:
            for k in range(3):
                state[k] = copy.deepcopy(np.fliplr(state_[k]))
                for i in range(self.boardSize - 1):
                    for j in range(self.boardSize - 1):
                        state[k, i, j] = np.roll(np.flip(state[k, i, j],0), shift=1)
            return state

    def get_children_node_prior_value(self):
        children_node_prior_value = len(self.availables)*[0]
        availables = copy.deepcopy(self.availables)
        for i in range(len(availables)):
            self.changeChessStatus(availables[i])
            if self.isHaveGetBlock()==True:
                children_node_prior_value[i] = (0.6)
            self.regainChessStatus(availables[i])
        return children_node_prior_value



class ChessBoard(QWidget,NormalChessBoard):
    # boardSize这里应为4或者6  boardSize为4说明棋盘为4x4  boardSize为6说明棋盘为6x6
    def __init__(self,boardSize = 4,parent=None):
        super().__init__(parent)
        NormalChessBoard.__init__(self,boardSize)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)  # 鼠标变成手指形状

        # chessMap 用于存储棋子的位置
        # blockState 用于存储块占领情况
        self.chessLabel = [QLabel(self) for i in range(self.boardSize * (self.boardSize - 1) * 2)]
        self.blockLabel = [QLabel(self) for i in range((self.boardSize - 1) * (self.boardSize - 1))]

        # 将所有块的中心坐标 写入blockLocationList
        for i in range(boardSize - 1):
            for j in range(boardSize - 1):
                pix_x, pix_y = j * 100 + self.firstBlockLoaction, i * 100 + self.firstBlockLoaction
                self.blockLocationList.append((pix_x, pix_y))
                self.blockLabel[i * (boardSize - 1) + j].setGeometry(pix_x - 38, pix_y - 38, 76, 76)

        # 加载图片
        self.blue_h = QPixmap('img/blue_h.png')
        self.blue_h_now = QPixmap('img/blue_h_now.png')
        self.blue_s = QPixmap('img/blue_s.png')
        self.blue_s_now = QPixmap('img/blue_s_now.png')
        self.blue_block = QPixmap('img/blue_block.png')

        self.red_h = QPixmap('img/red_h.png')
        self.red_h_now = QPixmap('img/red_h_now.png')
        self.red_s = QPixmap('img/red_s.png')
        self.red_s_now = QPixmap('img/red_s_now.png')
        self.red_block = QPixmap('img/red_block.png')

        self.setCursor(Qt.PointingHandCursor)  # 鼠标变成手指形状

    def reSetChess(self):
        print("重新开始！")
        self.chessMap = np.zeros((self.boardSize-1, self.boardSize-1, 4))
        self.blockState = np.zeros((self.boardSize-1, self.boardSize-1))
        self.playRecord = []
        self.chessNumMap = np.zeros(self.boardSize*(self.boardSize-1)*2)
        self.isEnd = False
        # 1表示蓝方下  0表示红方下
        self.currentPlayer = 1
        self.step = 0
        self.availables = list(range(self.boardSize*(self.boardSize-1)*2))
        for chesslabel in self.chessLabel:
            chesslabel.clear()
        for blocklabel in self.blockLabel:
            blocklabel.clear()


    # 交替换手 其中包括检查游戏是否结束
    # showResult默然为0 表示弹出框来显示最后结果
    def alternate(self):
        # getBlockList存储的是被当前下棋者占有的块的编号
        getBlockList = self.isGetBlock()
        if getBlockList:
            for i in getBlockList:
                self.blockLabel[i].setPixmap(self.blue_block if self.isBuleDown() else self.red_block)

        # 假如list为空
        if not getBlockList:
            self.currentPlayer = -self.currentPlayer


    def mousePressEvent(self,e):
        # if self.isBegin == 0:
        #     reply = QMessageBox.question(self, '帮助', '请先开始对弈！',
        #                                  QMessageBox.Yes)
        #     return
        # if e.button() == Qt.LeftButton:
        #     x, y = e.x(), e.y()  # 鼠标坐标
        #     self.drawByMousePixel((x,y))
        pass

    # 画的时候直接给出鼠标的坐标
    def drawByMousePixel(self,location):
        chessNum = self.pixelToChessNum(location)
        # 说明鼠标点击到了对应的位置
        if chessNum!=-1:
            self.do_action(chessNum)

    # chessNum为棋盘中标号 type为绘制棋子的类型
    # 0表示正常
    # 1表示含有红色圈圈的表示最后一次下的棋子
    # 2表示半透明的棋子
    def drawByChessNum(self,chessNum):
        # 检查chessNum
        if chessNum < 0 and chessNum >= self.chessNumMap.size:
            print("drawByChessNum-------chessNum异常!!  当前chessNum：",chessNum)
            return
        # # 当x,y处于棋盘的范围之内，且该位置上没有棋子时，对其进行绘制
        if self.chessNumMap[chessNum] == 0:
            pix_x, pix_y = self.chessLocationPixList[chessNum]
            # 改变棋盘状态信息chessMap
            self.changeChessStatus(chessNum)
            if self.isBuleDown():
                self.chessLabel[self.step].setPixmap(self.blue_h_now if self.chessDirection[chessNum] == 0 else self.blue_s_now)
            else:
                self.chessLabel[self.step].setPixmap(self.red_h_now if self.chessDirection[chessNum] == 0 else self.red_s_now)

            # 将上一步的棋子样式修改为normal状态
            if self.step >= 1:
                player , chessNum_ = self.playRecord[self.step-1]
                if player == 1:
                    self.chessLabel[self.step - 1].setPixmap(
                        self.blue_h if self.chessDirection[chessNum_] == 0 else self.blue_s
                    )
                else:
                    self.chessLabel[self.step - 1].setPixmap(
                        self.red_h if self.chessDirection[chessNum_] == 0 else self.red_s
                    )

            # 假如是横向的话
            if self.chessDirection[chessNum] == 0:
                self.chessLabel[self.step].setGeometry(pix_x - 40, pix_y - 10, 80, 20)
            else:
                self.chessLabel[self.step].setGeometry(pix_x - 10, pix_y - 40, 20, 80)
            self.step += 1

    def do_action(self,action,showDialog=0,autoJudge=1,autoReset=1):
        self.drawByChessNum(action)
        self.alternate()
        # 下面是判断输赢 autoJudge为1表示自动判断
        if autoJudge != 1:
            return
        isEnd, winner, blueGetBlock, redGetBlock = self.gameIsEnd()
        if isEnd:
            # print(self.blockState)
            # print(self.chessNumMap)
            # print(self.chessMap)

            if blueGetBlock > redGetBlock:
                winner = 1
                if showDialog:
                    reply = QMessageBox.question(self, '比赛结果',
                                                 '蓝方胜出！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock),
                                                 QMessageBox.Yes)
                print('GUI 蓝方胜出！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
            elif redGetBlock > blueGetBlock:
                winner = -1
                if showDialog:
                    reply = QMessageBox.question(self, '比赛结果!',
                                                 '红方胜出！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock),
                                                 QMessageBox.Yes)
                print('GUI 红方胜出！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock))
            else:
                winner = 0
                if showDialog:
                    reply = QMessageBox.question(self, '比赛结果!',
                                                 '平局！！蓝 {} : 红 {}   '.format(blueGetBlock, redGetBlock),
                                                 QMessageBox.Yes)
                print("GUI 平局")
            if autoReset:
                self.reSetChess()


    # 因为其父类是QWidget 没法deepcopy，这里提供一个方法转为普通的Board
    def to_normal_board(self):
        normal_board = NormalChessBoard(self.boardSize)
        normal_board.isEnd = self.isEnd
        normal_board.chessNumMap = self.chessNumMap
        normal_board.chessMap = self.chessMap
        normal_board.blockState = self.blockState
        normal_board.blockLocationList = self.blockLocationList
        normal_board.availables = self.availables
        normal_board.chessNumToMap = self.chessNumToMap
        normal_board.playRecord = self.playRecord
        normal_board.chessNumToBlockNum = self.chessNumToBlockNum
        normal_board.chessLocationPixList = self.chessLocationPixList
        normal_board.chessDirection = self.chessDirection
        normal_board.currentPlayer = self.currentPlayer
        normal_board.step = self.step
        normal_board.isBegin = self.isBegin
        return (copy.deepcopy(normal_board))


