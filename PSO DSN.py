from sklearn.datasets import load_iris, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from time import time
from operator import itemgetter
import numpy as np
import pandas as pd
import math

def data():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    mean_on_train = X_train.mean(axis=0)
    std_on_train = X_train.std(axis=0)
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    X_test_scaled = (X_test - mean_on_train) / std_on_train
    return X_train_scaled, X_test_scaled, y_train, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def myrange(start, end, step):
    r = start
    while (r <= end):
        yield r
        r += step

class DSN:
    def __init__(self, X_train, X_test, y_train, y_test, particle):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.particle = particle

    def module_by_module(self, input):
        global Om
        self.M = self.particle[0]
        concatenated_input = input
        for i in range(self.M):
            Wm = self.particle[2][i*2]
            Um = self.particle[2][i*2+1]
            Hm = sigmoid(np.dot(concatenated_input, Wm))
            Om = sigmoid(np.dot(Hm, Um))
            concatenated_input = np.hstack((concatenated_input, Om))
        return Om

    def return_training_error(self):
        output = self.module_by_module(self.X_train)
        output = np.where(output>=0.5, 1, output)
        output = np.where(output<0.5, 0, output)
        real = self.y_train.reshape((-1, 1))
        a, b = real.shape
        temp = np.zeros([a,b])
        temp = np.where(np.equal(output, real)==True, 1, temp)
        err = 1-np.sum(temp)/a
        return err

    def return_test_acc(self):
        output = self.module_by_module(self.X_test)
        real = self.y_test.reshape((-1,1))
        output = np.where(output >= 0.5, 1, output)
        output = np.where(output < 0.5, 0, output)
        a, b = real.shape
        temp = np.zeros([a, b])
        temp = np.where(np.equal(output, real) == True, 1, temp)
        err = np.sum(temp) / a
        return err

class Particle:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = data()
        self.D1 = self.X_train.shape[1]
        self.M = 1
        self.C = 1
        self.L = 10
        self.particle = self.make_initial_particle()
        self.best_particle = self.particle
        self.best_M = self.particle[0]
        self.best_value = self.fitness()

    def make_initial_particle(self):
        weight_list = []
        velocity_list = []
        for i in range(self.M):
            Dm = self.D1 + self.C * i
            W_initial = np.random.randn(Dm, self.L)
            U_initial = np.random.randn(self.L, self.C)
            weight_list.extend([W_initial, U_initial])
            W_velocity_initial, U_velocity_initial = self.random_matrix(Dm, self.L)
            velocity_list.extend([W_velocity_initial, U_velocity_initial])
        particle=[self.M, self.L, weight_list, velocity_list]
        return particle

    def random_matrix(self, Dm, Lm):
        Wrand = np.random.normal(0, 0.5, (Dm, Lm))
        Urand = np.random.normal(0, 0.5, (Lm, self.C))
        return Wrand, Urand

    def update_M(self, w, cp, cg, global_best_particle):
        original_M = self.particle[0]
        A = self.best_M - original_M
        B = global_best_particle[0] - original_M

        new_M = round(w * original_M + cp * np.random.random() * A + cg * np.random.random() * B)
        if new_M <= 1 :
            new_M =2

        self.particle[0] = new_M
        if new_M > original_M:
            for i in range(new_M - original_M):
                Dm = self.D1 + self.C * (original_M + i)
                Wrand, Urand = self.random_matrix(Dm, self.L)
                Vwrand, Vurand = self.random_matrix(Dm, self.L)
                self.particle[2].extend([Wrand, Urand])
                self.particle[3].extend([Vwrand, Vurand])
        else:
            self.particle[2] = self.particle[2][:new_M * 2]
            self.particle[3] = self.particle[3][:new_M * 2]

    def update_velocity(self, w, cp, cg, global_best_particle):
        # if self.particle[0] < global_best_particle[0]:
        #     length = self.particle[0]
        # else:
        length = global_best_particle[0]

        for i in range(length):
            for j in range(2):
                Dm = self.D1 + self.C * i
                a, b = self.random_matrix(Dm, self.L)
                if j == 0: ma = a
                elif j == 1: ma = b
                best_minus_current = self.best_particle[3][i*2+j] - self.particle[3][i*2+j]
                global_minus_current = global_best_particle[3][i*2+j] - self.particle[3][i*2+j]
                self.particle[3][i*2+j] = w * self.particle[3][i*2+j] + cp * ma * best_minus_current + cg * ma* global_minus_current
                self.particle[3][i*2+j] = np.where(self.particle[3][i*2+j]>=0.5, 0.5, self.particle[3][i*2+j])
                self.particle[3][i*2+j] = np.where(self.particle[3][i*2+j]<=-0.5, -0.5, self.particle[3][i*2+j])

    def update_weight(self):
        for i in range(self.M):
            self.particle[2][i] += self.particle[3][i]
            self.particle[2][i] = np.where(self.particle[2][i]>=1, 1, self.particle[2][i])
            self.particle[2][i] = np.where(self.particle[2][i]<=-1, -1, self.particle[2][i])
        new_value = self.fitness()

        if new_value < self.best_value:
            self.best_particle = self.best_particle
            self.best_M = self.particle[0]
            self.best_value = new_value
        return (self.particle, self.best_value)

    def fitness(self):
        dive_into_DSN = DSN(self.X_train, self.X_test, self.y_train, self.y_test, self.particle)
        err = dive_into_DSN.return_training_error()
        return err

def pso(num_iter, num_particles, w, cp, cg):
    seed_p = Particle()
    particles = [seed_p]
    best_global_particle = seed_p.best_particle
    best_global_value = seed_p.best_value

    for _ in range(num_particles-1):
        p = Particle()
        if p.best_value < best_global_value:
            best_global_particle = p.best_particle
            best_global_value = p.best_value
        particles.append(p)

    for _ in range(num_iter):
        for particle in particles:
            #particle.update_M(w, cp, cg, best_global_particle)
            particle.update_velocity(w, cp, cg, best_global_particle)
            new_position, new_value = particle.update_weight()
            if new_value < best_global_value:
                best_global_particle = new_position
                best_global_value = new_value

    print("Training: ", 1-best_global_value)

    X_train, X_test, y_train, y_test = data()
    total = DSN(X_train, X_test, y_train, y_test, best_global_particle)
    acc = total.return_test_acc()
    print("Test:", acc)
    return 1-best_global_value, acc

if __name__=="__main__":
    file =[]
    train_list =[]
    test_list = []
    time_ist =[]
    for i in range(30):
        start = time()
        train, test = pso(500, 20, 0.4, 5, 5)
        end = time()-start
        train_list.append(train)
        test_list.append(test)
        time_list.append(end)
    trainacc = np.mean(np.array(train_list))
    testacc = np.mean(np.array(test_list))
    averagetime = np.mean(np.array(time_list))
    print("average train acc : ", trainacc)
    print("average test acc : ", testacc)
    print("average time acc : ", averagetime)
        #
        # totaldata = [num, trainacc, testacc, averagetime]
        # file.append(totaldata)
        # datas = pd.DataFrame(file)
        # datas.columns = ['Num', 'Train', 'Test', 'Time']
        # datas.to_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Result per particle num_3.csv')


