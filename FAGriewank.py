from math import exp
import numpy as np
from SwarmPackagePy import *
from time import time
from math import *
import pandas as pd
from deap import benchmarks

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

class fa(sw):
    """
    Firefly Algorithm
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, csi=1, psi=1,
                 alpha0=1, alpha1=0.1, norm0=0, norm1=0.1):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param csi: beta zero (default value is 1)
        :param psi: light absorption coefficient of the medium
        (default value is 1)
        :param alpha0: initial value of the free randomization parameter alpha
        (default value is 1)
        :param alpha1: final value of the free randomization parameter alpha
        (default value is 0.1)
        :param norm0: first parameter for a normal (Gaussian) distribution
        (default value is 0)
        :param norm1: second parameter for a normal (Gaussian) distribution
        (default value is 0.1)
        """

        super(fa, self).__init__()

        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

        Pbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]

        Gbest = Pbest

        start = time()
        for t in range(iteration):

            alpha = alpha1 + (alpha0 - alpha1) * exp(-t)

            for i in range(n):
                fitness = [function(x) for x in self.__agents]
                for j in range(n):
                    if fitness[i] > fitness[j]:
                        self.__move(i, j, t, csi, psi, alpha, dimension,
                                    norm0, norm1)
                    else:
                        self.__agents[i] += np.random.normal(norm0, norm1,
                                                             dimension)


            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)

            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)
        global fitnessresult, totaltime
        totaltime = time() - start
        fitnessresult = function(Gbest)


    def __move(self, i, j, t, csi, psi, alpha, dimension, norm0, norm1):

        r = np.linalg.norm(self.__agents[i] - self.__agents[j])
        d1 = self.__agents[i][0] - self.__agents[j][0]
        d2 = self.__agents[i][1] - self.__agents[j][1]
        d = round(sqrt((d1**2) + (d2**2)),2)
        beta = csi / (exp(psi*(d)))

        self.__agents[i] = self.__agents[j] + beta * (self.__agents[i] - self.__agents[j]) + \
                           alpha * exp(-t) * np.random.normal(norm0, norm1, dimension)

def fitness(x):
    return benchmarks.griewank(x)[0]

def myrange(start, end, step):
    r = start
    while (r <= end):
        yield r
        r += step

file = []
for j in myrange(0.1,0.9,0.1):
    for i in range(30):
        faresult = fa(20, fitness, -600, 600, 3, 1000, csi=1, psi = j)
        totaldata = [j, fitnessresult, totaltime]
        file.append(totaldata)
        data = pd.DataFrame(file)
        data.columns = ['light absorption value', 'fitnessresult', 'totaltime']
        data.to_csv('C:/Users/lvaid/skku/대학원/학부인턴/FAGriewankResult.csv')
        print("Total time is %s." % totaltime)
        print(fitnessresult)



