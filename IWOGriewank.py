import numpy as np
import operator
from time import *
import pandas as pd


def griewank(x):
    result = ((x[0])**2)/4000 + ((x[1])**2)/4000 - np.cos(x[0]/np.sqrt(1)) * np.cos(x[1]/np.sqrt(2)) + 1
    return result

class Seed:
    def __init__(self, currentSeeds, iterationNo):
        self.currentSeeds = currentSeeds
        self.itno = iterationNo

    def evaluateSeed(self):
        seeds_value = []
        for i in range(len(self.currentSeeds)):
            seedfitness = griewank(self.currentSeeds[i])
            seeds_value.append(seedfitness)

        self.best_value = min(seeds_value)
        self.worst_value= max(seeds_value)

        newseed = []

        for i in range(len(self.currentSeeds)):
            ratio = (seeds_value[i] - self.worst_value) / (self.best_value - self.worst_value)

            s = np.round(minOfRandSeed + (maxOfRandSeed - minOfRandSeed) * ratio)

            self.currentSigma = (((maxit - self.itno) / (maxit - 1)) ** beta) * (
                        Sinitial - Sfinal) + Sfinal

            for k in range(int(s)):
                newseedPosition = self.currentSeeds[i] + self.currentSigma * np.random.rand(dim)
                newseed.append(newseedPosition)

        self.currentSeeds = newseed
        new_seeds_value = []
        for i in range(len(self.currentSeeds)):
            seedfitness = griewank(self.currentSeeds[i])
            new_seeds_value.append(seedfitness)

        self.best_value = min(new_seeds_value)
        self.worst_value = max(new_seeds_value)

        if len(self.currentSeeds) > maxOfPopSize:
            self.currentSeeds = Seed.deleteseeds(self)

    # newseedvalue을 내림차순으로 정렬.
    # maxOfPopSize와 같아질 때까지 worst한 값들 지워나가기

    def deleteseeds(self):
        seedinfo = []
        while len(self.currentSeeds) == maxOfPopSize:
            for i in range(len(self.currentSeeds)):
                newvalue = griewank(self.currentSeeds[i])
                t = (self.currentSeeds[i], newvalue)
                seedinfo.append(t)

            def seed0(tt):
                return tt[0]
            def seed1(tt):
                return tt[1]
            seedinfo.sort(key=seed1, reverse=False)
            del seedinfo[-1]

        reseed = []
        for u in range(len(seedinfo)):
            ss = seedinfo[u][0]
            reseed.append(ss)

        return reseed

def IWO():
    # 초기 랜덤 seed 배정
    initialSeed = []
    for _ in range(numOfPopSize):
        initialposition = np.random.rand(dim) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
        initialSeed.append(initialposition)

    seed = Seed(initialSeed, 1)
    bestValues =[]
    for it in range(maxit):
        iterationseed = Seed(seed.currentSeeds, it+2)
        iterationseed.evaluateSeed()
        bestvalue = iterationseed.best_value
        bestValues.append(bestvalue)

    return min(bestValues)

if __name__ == "__main__":
    dim = 2
    MIN_RANGE = -600
    MAX_RANGE = 600
    maxit = 1000
    Sfinal = 0.001
    numOfPopSize = 20
    maxOfPopSize = 30
    minOfRandSeed = 0
    #beta = 2
    #Sinitial = 0.5
    #Sfinal = 0.001

    #numOfPopSize = 10
    #maxOfPopSize = 25

    #minOfRandSeed = 0  # Minimum number of seeds
    #maxOfRandSeed = 3  # Maximum number of seeds


    def myrange(start, end, step):
        r = start
        while (r <= end):
            yield r
            r += step

    file = []
    for beta in myrange(2,7,1):
        for Sinitial in myrange(0.4,0.6,0.05):
                for maxOfRandSeed in myrange(2,6,1):
                    for i in range(30):
                        start = time()
                        result = IWO()
                        total_time = time()-start
                        finaldata = [beta, Sinitial, maxOfRandSeed, result, total_time]
                        print(finaldata)
                        file.append(finaldata)
                        data= pd.DataFrame(file)
                        data.columns = ['beta', 'Initial Sigma', 'MaxOfRandSeed', 'Result', 'TotalTime']
                        data.to_csv('C:/Users/lvaid/skku/대학원/학부인턴/IWOGriewank.csv')