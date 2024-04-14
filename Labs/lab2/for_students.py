from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # 2.1.1 Calculate fitness for each individual in the population
    fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]

    # Sum up the fitness values of all individuals
    sum_up = sum(fitnesses)

    # 2.1.2 Random parent selection. >fitness>chance
    def roulette_selection(populacja, n_selection):
        selected_pop = []
        for _ in range(n_selection):
            rand = random.uniform(0, sum_up)  # random float for selecting better parents
            temp_sum = 0  # sum of fitnesses
            for j, individual in enumerate(populacja):
                temp_sum += fitnesses[j]
                if temp_sum >= rand:
                    selected_pop.append(individual)
                    break
        return selected_pop

    selected = roulette_selection(population, n_selection)

    # 2.1.3 Perform crossover to create children from selected parents
    def crossover(selected_popul):
        children = []
        for i in range(0, n_selection, 2):
            second_parent = selected_popul[i]
            first_parent = selected_popul[i + 1]
            comb_point = random.randint(0, len(selected_popul[i]))  # parent fitness division

            child = first_parent[:comb_point] + second_parent[comb_point:]
            children.append(child)
            child =  second_parent[:comb_point] + first_parent[comb_point:]
            children.append(child)
        return children

    children = crossover(selected)

    # 2.1.4 Mutate children by flipping random bit
    def mutation(children, pMut):
        for child in children:
            for j in child:
                rand_number = random.uniform(0, 1)
                if rand_number < pMut:
                    child[j] = not child[j]
        return children

    children = mutation(children, 0.01)

    # 2.1.5 Update population by selecting elite individuals
    index = np.argsort(fitnesses, 0)[::-1]  # getting indexes from best to worst
    elite = []
    for i in range(n_elite):
        elite.append(population[index[i]])


    population = elite + children

    # Track the best individual and its fitness
    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness

    # Append the best fitness value to the history
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
