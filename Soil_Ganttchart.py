# Importing the matplotlb.pyplot
import matplotlib.pyplot as plt
from TransformInitial import TransformState, oil_info
from PSO import pso
import pandas as pd
from time import time
import csv

class Ganttchart():
    def __init__(self, gene):
        self.gene = gene    # [ [2, 0, [8, 23, 57]], [14, 0, [151, 158, 169]]], ... ]
        self.num_of_berth =  len(self.gene)

    def convert_schedule(self):
        self.longest_time = 0
        for i in range(self.num_of_berth):
            for k in range(len(self.gene[i])):
                if self.gene[i][k][2][2] >= self.longest_time:
                    self.longest_time = self.gene[i][k][2][2]

    def make_gantt_chart(self):
        self.convert_schedule()
        # Declaring a figure "gnt"
        fig, gnt = plt.subplots()

        # Setting X-axis & Y-axis limits
        gnt.set_ylim(0, self.num_of_berth)
        gnt.set_xlim(0, self.longest_time+10)

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('Hour')
        gnt.set_ylabel('Schedule per berth')

        # Setting y-axis
        gnt.set_yticks([15,25,35,45,55,65])
        gnt.set_yticklabels(['1', '2', '3', '4', '5'])

        # Setting graph attribute
        gnt.grid(True)

        # Declaring a color map
        cmap = plt.cm.Blues

        def drawLoadDuration(period, starty, opacity):
            gnt.broken_barh((period), starty, facecolors=cmap(opacity), lw=0, zorder=2)

        # Declaring a bar in schedule
        col = 0.1
        for i in range(self.num_of_berth):
            for k in range(len(self.gene[i])):
                schedule = []
                schedule.append((self.gene[i][k][2][1], self.gene[i][k][2][2]-self.gene[i][k][2][1]))     # [(34, 69)] 이런식으로 스케줄
                berth = ((self.gene[i][k][1]+1)*10, 10)          # 뒤에 있는 건 bar width 조정
                drawLoadDuration(schedule, berth, col)
                shipnum = self.gene[i][k][0]
                plt.text((schedule[0][0] + schedule[0][1]/2), (self.gene[i][k][1]+1)*10+5, shipnum, fontsize=16,
                         horizontalalignment='center', verticalalignment='center')
                plt.text((schedule[0][0]), (self.gene[i][k][1]+1)*10, self.gene[i][k][2][1], fontsize = 10,
                         horizontalalignment='center', verticalalignment='center')
                plt.text((schedule[0][0]+schedule[0][1]), (self.gene[i][k][1]+1)*10, self.gene[i][k][2][2], fontsize = 10,
                         horizontalalignment='center', verticalalignment='center')
                col+=0.05
        plt.show()

        
class Oilgraph():
    def __init__(self, oilschedule_x, oilschedule_y):
        # self.finish_time_list_for_oil_graph = 작업 종료 시간 리스트 = [ 0, 33, 33, 55, 55, 72, ...] 오일별로. 시간은 0 제외하고 이렇게 두번씩 들어가야 함!
        # self.oil_state_for_oil_graph= oil 리스트 = [ 200(0), 400(33), 300(33), 600(55), 200(55), ... ] oil 쓰기전, 쓴 후 결과만 반환
        self.oil_schedule_x = oilschedule_x
        self.oil_schedule_y = oilschedule_y

    def make_graph(self):
        fig = plt.figure(figsize=(20,8))
        for i in range(len(self.oil_schedule_x)):
            ax=fig.add_subplot(3, 4, i+1)
            ax.plot(self.oil_schedule_x[i], self.oil_schedule_y[i])
            ax.axhline(y=oil_info[i][0]*100/oil_info[i][1], color='r', linewidth=1)
            ax.set_xticks([0,100,200])
        plt.show()

if __name__ == "__main__":
    a = TransformState()
    # c = [[[2, 0, [8, 15, 49]], [9, 0, [98, 98, 131]], [15, 0, [152, 152, 183]]], [[6, 1, [74, 105, 131]], [12, 1, [140, 140, 194]]], [[1, 2, [2, 4, 39]], [3, 2, [31, 39, 71]], [8, 2, [86, 86, 122]], [11, 2, [134, 134, 151]], [13, 2, [151, 151, 162]], [17, 2, [184, 184, 200]]], [[0, 3, [0, 1, 13]], [5, 3, [37, 37, 123]], [10, 3, [108, 123, 137]], [14, 3, [151, 151, 162]], [16, 3, [156, 162, 173]]], [[4, 4, [36, 39, 53]], [7, 4, [77, 77, 136]]]]
    c =[[[5, 0, [37, 37, 123]], [17, 0, [184, 184, 200]]], [[2, 1, [8, 8, 42]], [4, 1, [36, 42, 56]], [8, 1, [86, 86, 122]], [10, 1, [108, 122, 136]], [12, 1, [140, 140, 194]]], [[0, 2, [0, 3, 15]], [9, 2, [98, 98, 131]], [15, 2, [152, 152, 183]]], [[1, 3, [2, 6, 41]], [3, 3, [31, 41, 73]], [7, 3, [77, 77, 136]], [11, 3, [134, 136, 153]], [13, 3, [151, 153, 164]], [14, 3, [151, 164, 175]], [16, 3, [156, 175, 186]]], [[6, 4, [74, 86, 112]]]]
    # c=[[[2, 0, [8, 10, 44]], [7, 0, [77, 77, 136]], [15, 0, [152, 152, 183]]], [[0, 1, [0, 10, 22]], [3, 1, [31, 31, 63]], [6, 1, [74, 74, 100]], [9, 1, [98, 100, 133]], [13, 1, [151, 151, 162]]], [[1, 2, [2, 2, 37]], [5, 2, [37, 37, 123]], [12, 2, [140, 140, 194]]], [[4, 3, [36, 55, 69]], [8, 3, [86, 86, 122]], [10, 3, [108, 122, 136]], [14, 3, [151, 151, 162]], [17, 3, [184, 184, 200]]], [[11, 4, [134, 134, 151]], [16, 4, [156, 156, 167]]]]

    print(a.countDemurrageTime(c))

    cc = a.check_oil_state(c)
    b = a.countDemurrageTime(cc)
    print(b)

    # Make gantt chart
    # d = Ganttchart(cc)
    # d.make_gantt_chart()

    # make oil schedule
    oilschedule_x, oilschedule_y = a.return_oil_for_oil_graph()
    oil = Oilgraph(oilschedule_x, oilschedule_y)
    oil.make_graph()

