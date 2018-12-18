import math
import random
import copy

import matplotlib.pyplot as plt
import numpy as np

from ga import GeneticAlgorithm


class TravellingSalesmanProblem(GeneticAlgorithm):
    """用GA算法模拟旅行商问题"""

    def __init__(self, genes, distance_matrix, visual=None):
        self.distance_matrix = distance_matrix
        super(TravellingSalesmanProblem, self).__init__(genes)
        self.visual = visual
        self.L = 1
        self.tmp = []

    def _swap(self, l1, l2):
        s = random.randint(0, len(l1) - 1)
        e = random.randint(0, len(l1) - 1)
        if s == e:
            return
        elif s > e:
            s, e = e, s

        frac1 = [i for i in (l1[:s] + l1[e:]) if i not in l2[s:e]]
        frac2 = [i for i in l1 if i not in (frac1 + l2[s:e])]
        r1 = frac1 + l2[s:e] + frac2
        frac3 = [i for i in (l2[:s] + l2[e:]) if i not in l1[s:e]]
        frac4 = [i for i in l2 if i not in (frac1 + l1[s:e])]
        r2 = frac3 + l1[s:e] + frac4
        l1[:] = r1[:]
        l2[:] = r2[:]

    def cross(self):
        """交叉互换"""
        for i in range(len(self.group)):
            if self.p_cross > random.random():
                self._swap(self.group[i], self.group[i - 1])

    def mutate(self):
        """变异"""
        for individual in self.group:
            if self.p_mutation > random.random():
                # 随机交换位置
                a = random.randint(0, len(individual) - 1)
                b = random.randint(0, len(individual) - 1)
                individual[a], individual[b] = individual[b], individual[a]

    def copy(self, fitnesses):
        """选择复制"""
        fits = [i - min(fitnesses) + 1 for i in fitnesses]
        fits_sum = sum(fits)

        def select():
            thread = random.random() * fits_sum
            pre_f, cur_f = 0, 0
            # 轮盘赌选择 / 比例选择
            for i, f in enumerate(fits):
                cur_f += f
                if pre_f <= thread <= cur_f:
                    self.tmp.append(copy.deepcopy(self.group[i]))
                    break

        for i in range(len(fitnesses)):
            select()
        self.group = self.tmp
        fitnesses = self.fit()
        if max(fitnesses) < self.best_fitness:
            fit_min_index = fitnesses.index(min(fitnesses))
            self.group[fit_min_index][:] = self.best_individual[:]
        self.tmp = []

    def fit(self):
        """计算适配值"""
        fitnesses = []
        N = self.distance_matrix.shape[0]
        factor = 76.5 * self.L * np.sqrt(N)
        for individual in self.group:
            d = self.distance(individual)
            fitnesses.append(factor / d)
        return fitnesses

    def distance(self, seq):
        d = 0
        for i in range(len(seq)):
            d += self.distance_matrix[seq[i - 1]][seq[i]]
        return d


def generate_cities():
    n_cities = random.randint(10, 20)
    cities = np.random.uniform(0, 500, (n_cities, 2))
    name = [i for i in range(n_cities)]
    start = name[0]
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        re_coord = cities - cities[i]
        distance_matrix[i, :] = np.sqrt(np.sum(re_coord * re_coord, 1))
    return name, cities, start, distance_matrix


def read_cities(file_name=None):
    if file_name is None:
        return generate_cities()
    cities, name = [], []
    with open(file_name, 'r', encoding='utf-8') as f:
        line = f.readline()
        line = f.readline()
        while line:
            line = f.readline()
            items = line.split('|')
            if len(items) >= 3:
                cities.append([float(items[1]), float(items[2])])
                name.append(int(items[0]))
    cities = np.asarray(cities)
    n_cities = len(name)
    start = name[0]
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        re_coord = cities - cities[i]
        distance_matrix[i, :] = np.sqrt(np.sum(re_coord * re_coord, 1))
    return name, cities, start, distance_matrix


def min_rect_perimeter(locations):
    locations = np.asarray(locations)
    x1, y1 = np.min(locations, 0)
    x2, y2 = np.max(locations, 0)
    return 2 * (x2 - x1 + y2 - y1)


if __name__ == '__main__':

    # cities, local, start, distance_matrix = read_cities()
    cities, local, start, distance_matrix = read_cities('cities.txt')
    L = min_rect_perimeter(local)

    # with open('cities.txt', 'w', encoding='utf-8') as f:
    #     print('name|x|y', file=f)
    #     print('---|---|---', file=f)
    #     for i, city in enumerate(local):
    #         print('{:d}|{:.4f}|{:.4f}'.format(i, city[0], city[1]), file=f)

    # tsp = TravellingSalesmanProblem(list(cities), distance_matrix, visual=True)
    # tsp.L = L
    # tsp.steps = 10000
    # individual, fitness = tsp.evolve()
    # print(individual, tsp.distance(individual))

    # ind = individual.index(start)
    # individual = individual[ind:] + individual[:ind]
    # seq = local[individual]
    # # 画点
    # plt.plot(np.array(seq[:, 0]), np.array(seq[:, 1]), 'b*')
    # for name, point in zip(individual, seq):
    #     plt.text(point[0] + 1, point[1] + 1, str(name))
    # plt.show()
    # for name, point in zip(individual, seq):
    #     plt.text(point[0] + 1, point[1] + 1, str(name))
    # seq = np.vstack((seq, [seq[0]]))
    # plt.plot(seq[:, 0], seq[:, 1], 'b-', marker='*', linewidth=1)
    # plt.show()

    individuals, fitnesses, distance = [], [], []
    for i in range(20):
        tsp = TravellingSalesmanProblem(list(cities), distance_matrix)
        tsp.L = L
        tsp.steps = 10000
        individual, fitness = tsp.evolve()
        ind = individual.index(start)
        individual = individual[ind:] + individual[:ind]
        individuals.append(individual)
        fitnesses.append(fitness)
        distance.append(tsp.distance(individual))

    with open('results.txt', 'w', encoding='utf-8') as f:
        print('||路径|适配值|距离', file=f)
        print('---|---|---|---', file=f)
        for i, (indiv, fit, d) in enumerate(
                zip(individuals, fitnesses, distance)):
            print('{:d}|{}|{:.6f}|{:.6f}'.format(i + 1, indiv, fit, d), file=f)
    i_min = fitnesses.index(min(fitnesses))
    i_max = fitnesses.index(max(fitnesses))
    print(f'Min-> path: {individuals[i_min]}, '
          f'fitnesses: {fitnesses[i_min]}, '
          f'distance: {distance[i_min]}')
    print(f'Max-> path: {individuals[i_max]}, '
          f'fitnesses: {fitnesses[i_max]}, '
          f'distance: {distance[i_max]}')
    print(
        f'Avr-> fitnesses: {np.mean(fitnesses)}, distance: {np.mean(distance)}'
    )
    print(
        f'Var-> fitnesses: {np.var(fitnesses)}, distance: {np.var(distance)}')
