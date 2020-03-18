from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from SwarmPackagePy import *
from deap import benchmarks
from math import exp
import numpy as np
from random import random
from time import time
import pandas as pd

class sw(object):
    def __init__(self):
        self.__Positions = []
        self.__Gbest = []

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""
        return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""
        return list(self.__Gbest)

class ba(sw):
    def __init__(self, n, function, lb, ub, dimension, iteration, r0=0.9,
                 V0=0.5, fmin=0, fmax=0.02, alpha=0.9, csi=0.9):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param r0: level of impulse emission (default value is 0.9)
        :param V0: volume of sound (default value is 0.5)
        :param fmin: min wave frequency (default value is 0)
        :param fmax: max wave frequency (default value is 0.02)
            fmin = 0 and fmax =0.02 - the bests values
        :param alpha: constant for change a volume of sound
         (default value is 0.9)
        :param csi: constant for change a level of impulse emission
         (default value is 0.9)
        """
        super(ba, self).__init__()

        r = [r0 for i in range(n)]

        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

        velocity = np.zeros((n, dimension))
        V = [V0 for i in range(n)]

        Pbest = self.__agents[np.array([function(i)
                                        for i in self.__agents]).argmin()]
        Gbest = Pbest

        f = fmin + (fmin - fmax)

        start = time()

        for t in range(iteration):

            sol = self.__agents

            F = f * np.random.random((n, dimension))
            velocity += (self.__agents - Gbest) * F
            sol += velocity

            for i in range(n):
                if random() > r[i]:
                    sol[i] = Gbest + np.random.uniform(-1, 1, (
                        1, dimension)) * sum(V) / n

            for i in range(n):
                if function(sol[i]) < function(self.__agents[i]) \
                        and random() < V[i]:
                    self.__agents[i] = sol[i]
                    V[i] *= alpha
                    r[i] *= (1 - exp(-csi * t))

            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)

            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)

        global fitnessresult, totaltime
        fitnessresult = fitness(Gbest)
        totaltime = time() - start

def fitness(x):
    return benchmarks.griewank(x)[0]

def myrange(start, end, step):
    r = start
    while (r <= end):
        yield r
        r += step

file = []
for l in myrange(0.7,0.9,0.1):
    for t in myrange(0.7,0.9,0.1):
        for j in myrange(0.7,0.9,0.1):
            for k in myrange(0.3,0.5,0.1):
                for i in range(30):
                    faresult = ba(20, fitness, -600, 600, 3, 1000, r0=j, V0=k, fmin=0, fmax=0.02, alpha=l, csi=t)
                    totaldata = [j, k, l, t, fitnessresult, totaltime]
                    file.append(totaldata)
                    data = pd.DataFrame(file)
                    data.columns = ['pulse rate', 'loudness', 'constant for loudness', 'constant for pulse', 'fitnessresult', 'totaltime']
                    data.to_csv('C:/Users/lvaid/skku/대학원/학부인턴/BAGriewankResult.csv')
                    print("Total time is %s." % totaltime)
                    print(fitnessresult)