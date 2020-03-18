# coding=utf-8
from time import time
from random import randint, uniform, random
from TransformInitial import TransformState
from Initial_w_Random import FinalInitial
import pandas as pd

class Particle:                       # particle 하나 들어감
    def __init__(self, particle):      # [4.0, 1.0, 0.0, 2.0, 1.16667, 4.25, 1.333333, ... ] 이런 애들 들어감, 각자 best_position과 best_value(p_best) 업데이트됨
        self.particle = particle
        self.transform = TransformState()
        self.dim = self.transform.num_of_ship  # 18개 배, dim= num_of_ship
        self.best_position = self.particle
        self.best_value = self.fitness_with_no_oil()
        self.velocity = self.random_velocity()  # 처음 속도는 0이나 1

    def random_velocity(self):
        v = []
        for i in range(self.dim):
            v.append(uniform(-1,1))
        return v

    def return_schedule(self):
        a = self.transform.berthAllocation_PSO(self.particle)
        self.schedule = self.transform.newSchedule(a)
        return self.schedule

    def fitness_with_no_oil(self):
        self.schedule = self.return_schedule()
        overlap = self.transform.check_overlapped_schedule(self.schedule)
        demur = self.transform.countDemurrageTime(self.schedule)
        fit = overlap + demur
        return fit

    def update_velocity(self, w, cp, cg, g):
        for d in range(self.dim):
            rp = self.random_velocity()
            rg = self.random_velocity()
            self.velocity[d] = (w * self.velocity[d] +
                                cp*rp[d]*(self.best_position[d] - self.particle[d]) +
                                cg*rg[d]*(g[d] - self.particle[d]))

    def update_position(self):
        for d in range(self.dim):
            self.particle[d] += self.velocity[d]

        for i in range(self.dim):         # 부두 index 제한
            if self.particle[i] < 0.0:
                self.particle[i] = uniform(0,1)
            elif self.particle[i] > 5.0:
                self.particle[i] = 4 + uniform(0,1)

        new_value= self.fitness_with_no_oil()

        if new_value < self.best_value:
            self.best_position = self.particle
            self.best_value = new_value

        return (self.particle, self.best_value, self.schedule)

def pso(num_iter, num_particle, w, cp, cg):
    a = TransformState()
    b = FinalInitial(num_particle)
    initial = b.total_random()
    seed_p= Particle(a.transform_PSO(initial[0]))
    particles = [seed_p]
    best_global_position = seed_p.best_position         # PSO gene 완성
    best_global_value = seed_p.best_value
    best_schedule = seed_p.schedule

    for i in range(len(initial)):
        particle = a.transform_PSO(initial[i])
        p = Particle(particle)                      # PSO gene을 가진 객체 생성
        if p.best_value < best_global_value:
            best_global_position = p.best_position
            best_global_value = p.best_value
        particles.append(p)

    for _ in range(num_iter):
        for particle in particles:
            particle.update_velocity(w, cp, cg, best_global_position)       #업데이트 시작
            new_position, new_value, new_schedule = particle.update_position()
            if new_value < best_global_value:
                best_global_position = new_position
                best_global_value = new_value
                best_schedule = new_schedule
                continue
        print('', sep='\n')
        print(_)
        print(best_global_value)
        if best_global_value <= 67:
            break

    return (best_global_position, best_global_value, best_schedule)

def myrange(start, end, step):
    r = start
    while (r <= end):
        yield r
        r += step

if __name__ == "__main__":
    file = []
    for k in range(20):
        start = time()
        gb, gv, schedule= pso(num_iter=1000, num_particle = 20, w=0.1, cp=0.2, cg=0.2)
        finish = time() - start
        print(finish)
        file.append([gb, gv, schedule, finish])
        data = pd.DataFrame(file)
        data.columns = ['Global Best', 'Global Best Value', 'Schedule', 'Total Time']
        data.to_csv('C:/Users/lvaid/skku/Technometrics/Projects/S-Oil/Rawdata/실험.csv')
