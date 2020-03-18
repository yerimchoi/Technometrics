from time import time
import csv
from operator import itemgetter
import numpy as np
from random import *
import math
import pandas as pd

def read_data(data):
    for i in range(len(data)):
        a = len(data[i])
        for j in range(a):
            if data[i][j] == '':
                del data[i][j:]
                break
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

# ship_info[i]=[체선count시간, laytime, 유종index, 작업시간, 기름양]
ship_info=[]
with open("C:/Users/lvaid/skku/Technometrics/Projects/S-Oil/package/ship_info.csv", "r") as f:
    rdr=csv.reader(f)
    for line in rdr:
        ship_info.append(line)
read_data(ship_info)

# oil_info[i]=[min,max,기준일기름양,생산량]
oil_info=[]
with open("C:/Users/lvaid/skku/Technometrics/Projects/S-Oil/package/oil_info_w_rate.csv", "r") as f:
    for line in f.readlines():
        line = [float(x.replace("\n", "")) for x in line.split(",")]
        oil_info.append(line)

initial_csv = []
with open("C:/Users/lvaid/skku/Technometrics/Projects/S-Oil/Rawdata/initialvalue4.csv", "r") as f:
    rdr = csv.reader(f)
    for line in rdr:
        initial_csv.append(line)

class TransformState():     # gene 하나씩 바꿉니다
    def __init__(self):
        self.num_of_popsize = 100
        self.num_of_berth = 5  # # of berths = num_of_berth
        self.num_of_ship = 18  # # of ships = num_of_ship
        self.num_of_oil = len(oil_info)
        self.penalty = 0
        self.random_schedule_list = [[] for k in range(self.num_of_popsize)]
        self.ship_schedule_using_this_oil = [[] for i in range(len(oil_info))] # ship_schedule_using_this_oil[0]= 0번 oil을 사용하는 배들의 스케줄 리스트 [[3, 0, 31, 62, 94], ...]
        self.finish_time_list_for_oil_graph = [[] for i in range(self.num_of_oil)]
        self.oil_state_for_oil_graph = [[] for i in range(self.num_of_oil)]
        self.gene_oil = [[] for i in range(len(oil_info))]  # 해당 oil 사용하는 배들의 index, gene_oil[0]=[0,11,5,16]...

        for i in range(len(ship_info)):
            num = 0
            while num <= self.num_of_oil:
                if ship_info[i][2] == num:
                    self.gene_oil[num].append(i)
                    break
                else:
                    num += 1
                    continue
        # print(gene_oil)     #[[7, 11], [2, 3, 8, 12], [6, 14], [4, 5, 10], [9, 15], [1], [0], [16], [13], [17]]

    def transform_PSO(self, gene):    # 염색체 하나에 대해서, self.initialPop 같이  [ [2, 0, [8, 23, 57]], [14, 0, [151, 158, 169]]], ... ] 이런 애 들어오면,
        list = []
        for berth in range(self.num_of_berth):
            berthLength = len(gene[berth])
            for ship in range(berthLength):
                list.append([gene[berth][ship][0], gene[berth][ship][1] + (1 / berthLength) * ship])
        list.sort(key=itemgetter(0))

        list2 = []
        for listnum in range(len(list)):
            list2.append(list[listnum][1])
        #print(list2)  #= [4.0, 1.0, 0.0, 2.0, 1.16667, 4.25, 1.333333, ... ] 이런 애들 나옴

        return list2

    def berthAllocation_PSO(self, gene):
        berthAllocationList = [[] for i in range(self.num_of_berth)]
        for berth in range(len(gene)):
            r = 1
            for num in range(self.num_of_berth):
                if gene[berth] // r == num:
                    berthAllocationList[num].append(gene.index(gene[berth]))
                    break
                else:
                    continue
        #print(berthAllocationList) # = [[7, 14], [5, 13], [1, 3, 10, 12, 15, 17], [2, 8], [0, 4, 6, 9, 11, 16]]

        return berthAllocationList

    def newSchedule(self, gene):
        # [[0, 1, 9, 15], [3, 6, 12, 16], [8, 14], [4, 7, 11, 13, 17], [2, 5, 10]] 이런식으로 들어옴

        self.newScheduleList = [[] for i in range(self.num_of_berth)]
        for berth in range(len(gene)):
            for ship in range(len(gene[berth])):
                shipindex = gene[berth][ship]
                demurCountTime = ship_info[shipindex][0]
                laytime = ship_info[shipindex][1]
                if ship == 0:
                    startTime = randint(ship_info[shipindex][0], ship_info[shipindex][0]+ship_info[shipindex][1])
                    finishTIme = startTime + ship_info[shipindex][3]
                    self.newScheduleList[berth].append([shipindex, berth, [demurCountTime, startTime, finishTIme]])
                else:
                    startTime = max(self.newScheduleList[berth][ship-1][2][2], ship_info[shipindex][0])
                    finishTIme = startTime + ship_info[shipindex][3]
                    self.newScheduleList[berth].append([shipindex, berth, [demurCountTime, startTime, finishTIme]])

        # 아래 check_overlapped_schedule & countDemurrageTime 에서 사용
        # print(self.newScheduleList) # = [[[0, 0, [0, 0, 12]], [3, 0, [31, 62, 94]], [5, 0, [37, 58, 144]], [13, 0, [151, 153, 164]], ..] 완성
        return self.newScheduleList

    def check_oil_state(self, gene):
        # gene = [[[0, 0, [0, 0, 12]], [3, 0, [31, 62, 94]], [5, 0, [37, 58, 144]]... 이렇게 생김
        for i in range(len(self.gene_oil)):
            for j in range(len(self.gene_oil[i])):
                self.ship_schedule_using_this_oil[i].append(self.find_ship_schedule(self.gene_oil[i][j], gene))
            self.ship_schedule_using_this_oil[i].sort(key=itemgetter(4))
        # print(self.ship_schedule_using_this_oil)    # [[[7, 1, 77, 78, 137], [11, 1, 134, 137, 154]], [[3, 2, 31, 45, 77], [2, 1, 8, 44, 78], ...]

        for i in range(len(self.ship_schedule_using_this_oil)):
            residual_oil = 0
            self.finish_time_list_for_oil_graph[i].append(0)
            self.oil_state_for_oil_graph[i].append(oil_info[i][2]*100/oil_info[i][1])
            totally_finish_time = self.ship_schedule_using_this_oil[i][0][4]

            for j in range(len(self.ship_schedule_using_this_oil[i])):
                ship = self.ship_schedule_using_this_oil[i][j]
                time_gap = 0
                process_time = ship_info[ship[0]][3]

                if self.ship_schedule_using_this_oil[i][j][4] > totally_finish_time:
                    totally_finish_time = self.ship_schedule_using_this_oil[i][j][4]

                if j == 0:
                    time_gap += ship[4]
                    residual_oil += oil_info[i][2] + oil_info[i][3] * time_gap - oil_info[i][4] * process_time
                else:
                    time_gap += ship[4] - self.ship_schedule_using_this_oil[i][j-1][4]
                    residual_oil += residual_oil + oil_info[i][3] * time_gap - oil_info[i][4] * process_time

                if residual_oil >= oil_info[i][1]:
                    residual_oil = oil_info[i][1]

                if residual_oil >= oil_info[i][0]:
                    if j ==0:
                        info = oil_info[i][2] + oil_info[i][3] * time_gap
                    else:
                        info = residual_oil + oil_info[i][3] * time_gap
                    if info >= oil_info[i][1]:
                        info = oil_info[i][1]
                    self.finish_time_list_for_oil_graph[i].extend([self.ship_schedule_using_this_oil[i][j][4], self.ship_schedule_using_this_oil[i][j][4]])
                    self.oil_state_for_oil_graph[i].extend([info*100/oil_info[i][1], residual_oil*100/oil_info[i][1]])

                elif residual_oil < oil_info[i][0]:         # oil_info[i][3] = 적재량
                    delay_by_oil = 2*abs(oil_info[i][0]-residual_oil)/oil_info[i][3]
                    print(self.ship_schedule_using_this_oil[i][j])
                    print(delay_by_oil)
                    ship[4] = ship[4] + delay_by_oil
                    self.ship_schedule_using_this_oil[i][j][4] = self.ship_schedule_using_this_oil[i][j][4] + delay_by_oil # 현재 배는 finish time만 뒤로 미룸
                    self.finish_time_list_for_oil_graph[i].extend(
                        [self.ship_schedule_using_this_oil[i][j][4]-delay_by_oil, self.ship_schedule_using_this_oil[i][j][4]])
                    self.oil_state_for_oil_graph[i].extend([(residual_oil + oil_info[i][3] * (ship[4]-ship[3]))*100/oil_info[i][1], oil_info[i][0]*100/oil_info[i][1]])

                    for k in range(len(self.ship_schedule_using_this_oil[i])-(j+1)):        # 그 다음 배는 작업 시작시간이랑 끝시간 모두 미룸
                        self.ship_schedule_using_this_oil[i][k+j+1][3] =  self.ship_schedule_using_this_oil[i][k+j+1][3] + delay_by_oil
                        self.ship_schedule_using_this_oil[i][k+j+1][4] =  self.ship_schedule_using_this_oil[i][k+j+1][4] + delay_by_oil

        #다시 gene로 반환
        list = [[] for i in range(self.num_of_berth)]
        for i in range(len(self.ship_schedule_using_this_oil)):
            for k in range(len(self.ship_schedule_using_this_oil[i])):
                for num in range(self.num_of_berth):
                    if self.ship_schedule_using_this_oil[i][k][1] == num:
                        list[num].append([self.ship_schedule_using_this_oil[i][k][0],self.ship_schedule_using_this_oil[i][k][1],
                                              [self.ship_schedule_using_this_oil[i][k][2],round(self.ship_schedule_using_this_oil[i][k][3]),round(self.ship_schedule_using_this_oil[i][k][4])]])
        return list

    def return_oil_for_oil_graph(self):
        # self.finish_time_list_for_oil_graph = 작업 종료 시간 리스트 = [ 0, 33, 33, 55, 55, 72, ...] 오일별로. 시간은 0 제외하고 이렇게 두번씩 들어가야 함!
        # self.oil_state_for_oil_graph= oil 리스트 = [ 200(0), 400(33), 300(33), 600(55), 200(55), ... ] oil 쓰기전, 쓴 후 결과만 반환
        print(self.finish_time_list_for_oil_graph)
        print(self.oil_state_for_oil_graph)
        return self.finish_time_list_for_oil_graph, self.oil_state_for_oil_graph

    def find_ship_schedule(self, index, gene):  #index와 gene을 받으면 그 index를 가진 배가 gene에서 어떤 스케줄 갖는지 반환. [3, 0, 31, 62, 94]
        for i in range(self.num_of_berth):
            for j in range(len(gene[i])):
                if gene[i][j][0]== index:
                    return [gene[i][j][0], gene[i][j][1], gene[i][j][2][0], gene[i][j][2][1], gene[i][j][2][2]]

    def check_overlapped_schedule(self, gene):      # 하나의 염색체에 대해서, 겹치는 스케줄에 대해 big M 부여
        value = 0
        for berth in range(len(gene)):
            for ship in range(len(gene[berth])-1):
                set1= {i for i in range(gene[berth][ship][2][1], gene[berth][ship][2][2])}
                set2= {i for i in range(gene[berth][ship+1][2][1], gene[berth][ship+1][2][2])}
                if len(list(set1 & set2)) >1:
                    value += 1000 * len(list(set1 & set2))
        # print(value)
        return value

    def countDemurrageTime(self,gene):        # 하나의 염색체에 대해서, demurrage time 계산 = objective function
        demurrage = 0
        for berth in range(len(gene)):
            for ship in range(len(gene[berth])):
                value = gene[berth][ship][2][2]-(gene[berth][ship][2][0]+ship_info[gene[berth][ship][0]][1])
                if value >= 0 :
                    demurrage += value
        # print(demurrage)
        return demurrage

if __name__ == "__main__":
    for i in range(20):
        particle =[2.6398721278729016, 3.301441550007404, 1.6869082991673263, 2.659065964113016, 0.9139361528096991, 4.0952890876737165, 3.3314856232540033, 1.2105863928284606, 2.6946490789991553, 3.030521283970387, 2.0937137428587222, 1.4909051437919865, 0.8682368378049976, 1.126968636249705, 1.4355067044219318, 2.0460726980918524, 4.3943363757829434, 4.0874569600275885]
        a = TransformState()
        b=a.berthAllocation_PSO(particle)
        c= a.newSchedule(b)
        print(a.countDemurrageTime(c))  #112 시간
        d =a.check_oil_state(c)
        # print(d)
        # print(a.countDemurrageTime(d))




