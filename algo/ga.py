import logging
import numpy as np
from datetime import datetime
from typing import List, Dict
from .server_placer import ServerPlacer
from data.edge_server import EdgeServer

class GAServerPlacer(ServerPlacer):
    """
    Genetic Algorithm (GA) approach for edge server placement.
    This approach optimizes workload balancing and communication delay minimization simultaneously.
    """
    name = 'GA'

    def __init__(self, base_stations, distances, population_size=30, max_generations=100, mutation_rate=0.1, crossover_rate=0.9):
        super().__init__(base_stations, distances)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_base_stations = len(base_stations)
        self.num_edge_servers = len(distances)
        
        # Initialize population: random placements of base stations on edge servers
        self.population = np.random.rand(self.population_size, self.num_base_stations)
        self.fitness_values = np.inf * np.ones(self.population_size)  # Fitness for each individual
        
        # Define weight factors for objectives
        self.alpha = 0.5  # Weight factor for workload balancing
        self.beta = 0.5   # Weight factor for communication delay minimization

    def compute_objectives(self):
        """
        Compute the objectives for workload balancing and communication delay minimization.
        """
        workload_balances = []
        communication_delays = []

        for i in range(self.population_size):
            individual = self.population[i]

            # Calculate workload balance and communication delay
            workload_balance = self._calculate_workload_balance(individual)
            communication_delay = self._calculate_communication_delay(individual)

            # Append to lists
            workload_balances.append(workload_balance)
            communication_delays.append(communication_delay)

        return np.array(workload_balances), np.array(communication_delays)

    def _calculate_workload_balance(self, individual):
        """
        Compute the workload balance objective for the current individual.
        """
        total_workload = np.sum([bs.workload for bs in self.base_stations])
        edge_server_workloads = np.zeros(self.num_edge_servers)

        # Assign base stations to edge servers based on individual (placement) and calculate workloads
        for i, base_station in enumerate(self.base_stations):
            closest_edge_server_idx = int(individual[i] * self.num_edge_servers)
            edge_server_workloads[closest_edge_server_idx] += base_station.workload

        # Normalize workloads and compute the balance metric
        max_workload = np.max(edge_server_workloads)
        min_workload = np.min(edge_server_workloads)
        balance = max_workload - min_workload  # Smaller value is better

        return balance

    def _calculate_communication_delay(self, individual):
        """
        Compute the communication delay objective for the current individual.
        """
        delay = 0
        for i, base_station in enumerate(self.base_stations):
            closest_edge_server_idx = int(individual[i] * self.num_edge_servers)
            delay += self.distances[i][closest_edge_server_idx] * base_station.workload
        
        return delay

    def _fitness_function(self, i):
        """
        Compute the fitness value for an individual.
        """
        workload_balance, communication_delay = self.compute_objectives()
        fitness = self.alpha * workload_balance[i] + self.beta * communication_delay[i]
        return fitness

    def selection(self):
        """
        Tournament selection for selecting parents based on fitness.
        """
        selected_parents = []
        for _ in range(self.population_size):
            # Tournament selection: pick two random individuals, select the best one
            tournament = np.random.choice(self.population_size, 2, replace=False)
            winner = tournament[0] if self.fitness_values[tournament[0]] < self.fitness_values[tournament[1]] else tournament[1]
            selected_parents.append(self.population[winner])
        return np.array(selected_parents)

    def crossover(self, parents):
        """
        One-point crossover between pairs of parents to produce offspring.
        """
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.num_base_stations)
                offspring[i, :crossover_point] = parents[i, :crossover_point]
                offspring[i+1, crossover_point:] = parents[i+1, crossover_point:]
                offspring[i+1, :crossover_point] = parents[i+1, :crossover_point]
                offspring[i, crossover_point:] = parents[i, crossover_point:]
            else:
                offspring[i] = parents[i]
                offspring[i+1] = parents[i+1]
        return offspring

    def mutation(self, offspring):
        """
        Mutation: Randomly change positions with a certain probability (mutation_rate).
        Ensures mutation point is within bounds of the individual array.
        """
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(self.num_base_stations)  # Ensure it stays within bounds
                # mutation_point = np.random.choice(self.num_base_stations)  # Selects an index within valid range
                print(f"Mutation Point: {mutation_point} (valid range: 0 to {self.num_base_stations - 1})")  # Debug
                offspring[i, mutation_point] = np.random.rand()  # Randomly change the position
        return offspring

    def replace_population(self, offspring):
        """
        Replace the current population with the offspring.
        """
        self.population = offspring

    def update_population(self):
        """
        Update the population by evaluating fitness, selecting parents, performing crossover and mutation, and replacing the old population.
        """
        # Evaluate fitness for the current population
        for i in range(self.population_size):
            self.fitness_values[i] = self._fitness_function(i)

        # Selection
        selected_parents = self.selection()

        # Crossover
        offspring = self.crossover(selected_parents)

        # Mutation
        offspring = self.mutation(offspring)

        # Replace population with offspring
        self.replace_population(offspring)

    def place_server(self, base_station_num, edge_server_num):
        """
        Main function to place servers using GA.
        """
        logging.info("{0}: Start running GA with N={1}, K={2}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                      base_station_num, edge_server_num))
        self.num_base_stations = base_station_num
        self.num_edge_servers = edge_server_num

        for generation in range(self.max_generations):
            self.update_population()

            # Best solution (global best) in this generation
            best_idx = np.argmin(self.fitness_values)
            logging.info(f"Generation {generation + 1}/{self.max_generations} - Best Fitness: {self.fitness_values[best_idx]}")

        # After finding the best placement, assign base stations to edge servers
        best_individual = self.population[best_idx]
        edge_servers = [EdgeServer(i, self.base_stations[i].latitude, self.base_stations[i].longitude) for i in range(edge_server_num)]
        for i, base_station in enumerate(self.base_stations):
            closest_edge_server_idx = int(best_individual[i] * edge_server_num)
            edge_servers[closest_edge_server_idx].assigned_base_stations.append(base_station)
            edge_servers[closest_edge_server_idx].workload += base_station.workload

        self.edge_servers = edge_servers
        logging.info("{0}: End running GA".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
