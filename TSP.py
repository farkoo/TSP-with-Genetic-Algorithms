# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:24:25 2023

@author: Farkoo
"""

import math
import random
import numpy as np


class TSPSolver:
    def __init__(self, 
                 city_coordinates=[],
                 population_size=100, 
                 mutation_rate=0.05, 
                 crossover_rate=0.8, 
                 num_generations=200, 
                 crossover_type='ordered',
                 survivor_selection_type='plus'):
        
        self.city_coordinates = city_coordinates
        self.num_cities = len(city_coordinates)
        self.city_coordinates = city_coordinates
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.crossover_type = crossover_type
        self.survivor_selection_type = survivor_selection_type

    def calculate_distance(self, city1, city2):
        x1, y1 = city1
        x2, y2 = city2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def initialize_population(self, population_size):
        population = []
        for _ in range(population_size):
            tour = list(range(1, self.num_cities + 1))
            random.shuffle(tour)
            population.append(tour)
        return population

    def evaluate_fitness(self, population):
        fitness_scores = []
        for tour in population:
            distance = 0
            for i in range(len(tour) - 1):
                city1 = self.city_coordinates[tour[i] - 1]
                city2 = self.city_coordinates[tour[i + 1] - 1]
                distance += self.calculate_distance(city1, city2)
            fitness_scores.append(1 / distance)  # Inverse of the distance as fitness score
        return fitness_scores

    def tournament_selection(self, population, fitness_scores, tournament_size):
        selected_parents = []
        for _ in range(len(population)):
            tournament = random.sample(range(len(population)), tournament_size)
            selected_parent = max(tournament, key=lambda x: fitness_scores[x])
            selected_parents.append(population[selected_parent])
        return selected_parents

    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a

        child = [-1] * size
        child[a:b + 1] = parent1[a:b + 1]

        remaining = [x for x in parent2 if x not in child]
        child[:a] = remaining[:a]
        child[b + 1:] = remaining[a:]

        return child

    def cycle_crossover(self, parent1, parent2):
        size = len(parent1)
        child1 = [-1] * size
        child2 = [-1] * size

        # Select a random starting point
        start_index = random.randint(0, size - 1)
        cycle = []

        # Perform cycle crossover
        while True:
            cycle.append(start_index)
            value_parent2 = parent2[start_index]
            index_parent1 = parent1.index(value_parent2)

            if index_parent1 in cycle:
                break

            start_index = index_parent1

        # Create offspring using the cycle
        for index in cycle:
            child1[index] = parent1[index]
            child2[index] = parent2[index]

        # Fill in remaining positions with the values from the other parent
        for i in range(size):
            if child1[i] == -1:
                child1[i] = parent2[i]
            if child2[i] == -1:
                child2[i] = parent1[i]

        return child1, child2

    def swap_mutation(self, tour, mutation_rate):
        for i in range(len(tour)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(tour) - 1)
                tour[i], tour[j] = tour[j], tour[i]
        return tour

    def select_best_individuals(self, population, offspring, fitness_scores, offspring_fitness_scores):
        if self.survivor_selection_type == 'plus':
            combined_population = population + offspring
            combined_fitness_scores = fitness_scores + offspring_fitness_scores
        elif self.survivor_selection_type == 'comma':
            combined_population = offspring
            combined_fitness_scores = offspring_fitness_scores
        sorted_population = [x for _, x in sorted(zip(combined_fitness_scores, combined_population), reverse=True)]
        return sorted_population[:len(population)]

    def solve_tsp(self):
        # Step 3: Generate an initial population of candidate solutions
        population = self.initialize_population(self.population_size)

        for generation in range(self.num_generations):
            # Step 4: Evaluate the fitness of each solution in the population
            fitness_scores = self.evaluate_fitness(population)

            # Print the best fitness score in the current generation
            best_fitness = max(fitness_scores)
            print(f"Generation {generation+1}: Best fitness = {best_fitness}")

            # Step 5a: Select parents from the current population
            parents = self.tournament_selection(population, fitness_scores, tournament_size=3)

            offspring = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # Step 5b: Apply crossover to create offspring
                if random.random() < self.crossover_rate:
                    if self.crossover_type == 'ordered':
                        child1 = self.ordered_crossover(parent1, parent2)
                        child2 = self.ordered_crossover(parent2, parent1)
                    elif self.crossover_type == 'cycle':
                        child1, child2 = self.cycle_crossover(parent1, parent2)
                else:
                    child1 = parent1
                    child2 = parent2

                # Step 5c: Apply mutation to the offspring
                child1 = self.swap_mutation(child1, self.mutation_rate)
                child2 = self.swap_mutation(child2, self.mutation_rate)

                offspring.extend([child1, child2])

            # Step 5d: Evaluate the fitness of the new offspring
            offspring_fitness_scores = self.evaluate_fitness(offspring)

            # Step 5e: Select the best individuals for the next generation
            population = self.select_best_individuals(population, offspring, fitness_scores, offspring_fitness_scores)

        # Select the best solution from the final population
        best_solution = population[0]

        return best_solution


def read_city_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_index = lines.index("NODE_COORD_SECTION\n") + 1
        end_index = lines.index("EOF\n")
        city_data = [list(map(float, line.strip().split()[1:])) for line in lines[start_index:end_index]]
        city_coordinates = np.array(city_data)
        return city_coordinates

def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        line = file.readline()
        numbers = list(map(int, line.strip().split()))
    return numbers


# Main program
if __name__ == '__main__':
    input_file = 'wi29.tsp'  # Replace with the actual file path
    city_coordinates = read_city_coordinates(input_file)
    
    tsp_solver = TSPSolver(city_coordinates,
                            population_size=100, 
                            mutation_rate=0.05,     
                            crossover_rate=0.8, 
                            num_generations=600, 
                            crossover_type='ordered',
                            survivor_selection_type='plus')
    
    best_solution = tsp_solver.solve_tsp()
    print("Best tour:", *best_solution, best_solution[0], sep=' ')
    fitness = 1 / tsp_solver.evaluate_fitness([best_solution])[0]
    print("Tour length:", fitness)  # Distance of the best tour
    
    with open('out_' + input_file, 'w') as file:
        print(*best_solution, best_solution[0], sep=' ', file=file)
        
    file_path = 'out_wi29.tsp'  # Replace with the actual file path
    tour = read_numbers_from_file(file_path)
    print(tour)
    # tsp_solver = TSPSolver(city_coordinates=city_coordinates)
    fitness = 1 / tsp_solver.evaluate_fitness([tour])[0]
    print("Tour length:", fitness)
