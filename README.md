# TSP Solver

This code provides a solver for the Traveling Salesman Problem (TSP). The TSP is a classic optimization problem where the goal is to find the shortest possible route that visits a set of cities and returns to the starting city.

## Dependencies

The code requires the following dependencies:

- `math`: A module providing mathematical functions.
- `random`: A module for generating random numbers.
- `numpy`: A library for handling arrays and numerical operations.

## Usage

### Initializing the Solver

To use the `TSPSolver` class, create an instance of it with the desired parameters:

```python
tsp_solver = TSPSolver(
    city_coordinates=[(x1, y1), (x2, y2), ...],
    population_size=100,
    mutation_rate=0.05,
    crossover_rate=0.8,
    num_generations=200,
    crossover_type='ordered',
    survivor_selection_type='plus'
)
```

* `city_coordinates` (list of tuples): The coordinates of the cities in the TSP problem.
* `population_size` (int): The size of the population in each generation.
* `mutation_rate` (float): The probability of a mutation occurring during crossover.
* `crossover_rate` (float): The probability of crossover occurring between two parents.
* `num_generations` (int): The number of generations to run the genetic algorithm.
* `crossover_type` (string): The type of crossover to use. Options are 'ordered' and 'cycle'.
* `survivor_selection_type` (string): The type of survivor selection to use. Options are 'plus' and 'comma'.

### Solving the TSP
To solve the TSP problem, call the solve_tsp() method on the TSPSolver instance:
```python
best_solution = tsp_solver.solve_tsp()
```

This method returns the best solution found, which is a list representing the order in which the cities should be visited.

### Additional Functions
The code provides two additional utility functions:

1. `read_city_coordinates(file_path)`
This function reads the city coordinates from a file and returns them as a NumPy array.
```python
city_coordinates = read_city_coordinates(file_path)
```
`file_path` (string): The path to the file containing the city coordinates.


2. `read_numbers_from_file(file_path)`
This function reads a list of numbers from a file and returns them as a list.
```python
numbers = read_numbers_from_file(file_path)
```
`file_path` (string): The path to the file containing the numbers.

## Running the Code
To run the code, follow these steps:

1. Set the input file path `(input_file)` to the file containing the city coordinates.
2. Set the output file path `(file_path)` to save the best tour.
3. Create an instance of the `TSPSolver` class with the desired parameters.
4. Call the `solve_tsp()` method to obtain the best solution.
5. Print the best tour and its length.
6. Save the best tour to the output file.

Make sure to replace the placeholder values with the actual file paths.

```python
import math
import random
import numpy as np

class TSPSolver:
    # Implementation of the TSPSolver class
    ...

def read_city_coordinates(file_path):
    # Implementation of the read_city_coordinates function
    ...

def read_numbers_from_file(file_path):
    # Implementation of the read_numbers_from_file function
    ...

# Main program
if __name__ == '__main__':
    input_file = 'wi29.tsp'  # Replace with the actual file path
    city_coordinates = read_city_coordinates(input_file)

    tsp_solver = TSPSolver(
        city_coordinates=city_coordinates,
        population_size=100,
        mutation_rate=0.05,
        crossover_rate=0.8,
        num_generations=600,
        crossover_type='ordered',
        survivor_selection_type='plus'
    )

    best_solution = tsp_solver.solve_tsp()
    print("Best tour:", *best_solution, best_solution[0], sep=' ')
    fitness = 1 / tsp_solver.evaluate_fitness([best_solution])[0]
    print("Tour length:", fitness)  # Distance of the best tour

    with open('out_' + input_file, 'w') as file:
        print(*best_solution, best_solution[0], sep=' ', file=file)

    file_path = 'out_wi29.tsp'  # Replace with the actual file path
    tour = read_numbers_from_file(file_path)
    print(tour)
    fitness = 1 / tsp_solver.evaluate_fitness([tour])[0]
    print("Tour length:", fitness)
```

## Support

**Contact me @:**

e-mail:

* farzanehkoohestani2000@gmail.com

Telegram id:

* [@farzaneh_koohestani](https://t.me/farzaneh_koohestani)

## License
[MIT](https://github.com/farkoo/TSP-with-Genetic-Algorithms/blob/master/LICENSE)
&#0169; 
[Farzaneh Koohestani](https://github.com/farkoo)
