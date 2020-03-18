from time import time
import csv
from operator import itemgetter
import numpy as np
from random import *
import math
import pandas as pd
from TransformInitial import TransformState

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

class FinalInitial:
    def __init__(self, num_particle):
        self.num_particle = num_particle
        self.num_FCFS_particle = round(num_particle/10)
        self.num_random_particle = num_particle-self.num_FCFS_particle
        self.num_of_ship = len(ship_info)
        self.num_of_berth = 5
        self.initial_fcfs_schedule = [[[] for j in range(5)] for i in
                                       range(len(initial_csv))]  # 처음 배 스케줄 따온 후 전 배가 작업 끝나면 바로 작업 시작
        self.transform = TransformState()

    def initial_FCFS_Schedule(self):  # FCFS로 일단 불러들여, InitialValue에서 생성한거
        initial_info = [[] for i in range(len(initial_csv))]  # initial 부두 index들만 담고 있는 리스트
        for i in range(len(initial_csv)):
            for j in range(5):
                a = initial_csv[i][j][1:-1].split(',')
                list = []
                for k in range(len(a)):
                    list.append(int(a[k]))
                initial_info[i].append(list)
        # print(initial_info[0])  # initial 불러오기 완료, [[2, 3, 8], [0, 9, 13, 16], [4, 5, 12], [1, 7, 14, 17], [6, 10, 11, 15]]

        initial_firstship_schedule = [[] for i in range(len(initial_csv))]  # initial 부두의 처음 배의 작업 시작시간, 종료 시간 등을 담고 있는 리스트
        for i in range(len(initial_csv)):
            list = []
            for j in range(5):
                b = int(initial_csv[i][j + 5][1:-1].split(',')[3])
                c = int(initial_csv[i][j + 5][1:-1].split(',')[4][:-1])
                list.append([b, c])
            initial_firstship_schedule[i] = list

        for i in range(len(initial_csv)):
            for j in range(5):
                for ship in range(len(initial_info[i][j])):
                    shipindex = initial_info[i][j][ship]
                    if ship == 0:
                        startTime = initial_firstship_schedule[i][j][0]
                        finishTime = initial_firstship_schedule[i][j][1]
                        self.initial_fcfs_schedule[i][j].append(
                            [shipindex, j, [ship_info[shipindex][0], startTime, finishTime]])
                    else:
                        startTime = max(ship_info[shipindex][0], self.initial_fcfs_schedule[i][j][ship - 1][2][2])
                        finishTime = startTime + ship_info[shipindex][3]
                        self.initial_fcfs_schedule[i][j].append(
                            [shipindex, j, [ship_info[shipindex][0], startTime, finishTime]])
        # print(initial_total_schedule[99]) # = [[[2, 0, [8, 9, 43]], [6, 0, [74, 74, 100]], [9, 0, [98, 100, 133]], [15, 0, [152, 152, 183]]] ...] 꼴로 출력
        return self.initial_fcfs_schedule

    def initial_w_random(self):
        total_initial = []
        fcfs = self.initial_FCFS_Schedule()
        for i in range(self.num_FCFS_particle):         # FCFS particle 생성
            total_initial.append(fcfs[i])
        for j in range(self.num_random_particle):       # 랜덤 particle 생성
            random_particle = [[] for i in range(self.num_of_berth)]
            for ship in range(self.num_of_ship):
                random_particle[randint(0,4)].append(ship)              #[[3, 7, 10, 11, 12], [4, 5, 8, 14], [0, 1, 2, 6, 9, 13, 15, 16, 17], [], []]
            particle = self.transform.newSchedule(random_particle)
            # print(particle)
            total_initial.append(particle)
        # print(total_initial)
        return total_initial

    # 현재 가장 성능이 좋은 전체 랜덤
    def total_random(self):
        total_random_initial = []
        for j in range(self.num_particle):  # totally 랜덤 particle 생성
            random_particle = [[] for i in range(self.num_of_berth)]
            for ship in range(self.num_of_ship):
                random_particle[randint(0, 4)].append(
                    ship)  # [[3, 7, 10, 11, 12], [4, 5, 8, 14], [0, 1, 2, 6, 9, 13, 15, 16, 17], [], []]
            particle = self.transform.newSchedule(random_particle)
            total_random_initial.append(particle)
        return total_random_initial


