import abc
import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np


class GeneticAlgorithm(object):
    """通过选择染色体交叉互换、变异实现最优化。"""

    def __init__(self, genes):

        self.__metaclass__ = abc.ABCMeta

        self.p_cross = 0.5
        self.p_mutation = 0.05
        self.popSize = 10
        self.steps = 10000
        self.group = []
        self.genes = genes

        self.best_individual = None
        self.best_fitness = None
        self.local_best_fitnesses = []
        self.best_fitnesses = []
        self.visual = None

    @abc.abstractmethod
    def cross(self):
        """交叉互换"""
        pass

    @abc.abstractmethod
    def mutate(self):
        """变异"""
        pass

    @abc.abstractmethod
    def copy(self, fitnesses):
        """复制"""
        pass

    @abc.abstractmethod
    def replace(self, fitnesses):
        """替换最差的个体"""
        pass

    @abc.abstractmethod
    def fit(self):
        """计算适配值"""
        pass

    def create_group(self):
        for i in range(self.popSize):
            self.group.append(random.sample(self.genes, len(self.genes)))

    def evolve(self):
        """进化"""

        step = 0

        # 初始化种群
        self.create_group()
        fitnesses = self.fit()

        best_fitness = max(fitnesses)
        best_index = fitnesses.index(best_fitness)
        self.best_fitness = best_fitness
        self.best_individual = copy.deepcopy(self.group[best_index])
        # 进化
        while step < self.steps:
            step += 1
            self.copy(fitnesses)
            self.cross()
            self.mutate()
            fitnesses = self.fit()
            if self.visual:
                self.local_best_fitnesses.append(best_fitness)
                self.best_fitnesses.append(self.best_fitness)
            best_fitness = max(fitnesses)
            if best_fitness > self.best_fitness:
                best_index = fitnesses.index(best_fitness)
                self.best_fitness = best_fitness
                self.best_individual = copy.deepcopy(self.group[best_index])

        if self.visual:
            n = np.linspace(1, step, step)
            plt.plot(n, self.best_fitnesses)
            plt.show()
            plt.plot(n, self.local_best_fitnesses)
            plt.show()

        return self.best_individual, self.best_fitness
