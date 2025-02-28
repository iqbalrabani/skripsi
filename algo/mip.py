import math
import logging
from datetime import datetime
from typing import List, Iterable

import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, value

from data.base_station import BaseStation
from data.edge_server import EdgeServer
from .server_placer import ServerPlacer


class MIPServerPlacer(ServerPlacer):
    """
    MIP approach using PuLP
    """
    name = 'MIP'

    def __init__(self, base_stations: List[BaseStation], distances: List[List[float]]):
        super().__init__(base_stations, distances)
        self.n = 0
        self.k = 0
        self.weights = None
        self.belongs = None
        self.assign = None

    def place_server(self, base_station_num, edge_server_num):
        logging.info("{0}:Start running MIP with N={1}, K={2}".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), base_station_num, edge_server_num))

        self.n = min(base_station_num, len(self.base_stations))
        self.k = edge_server_num

        self.preprocess_problem()

        # Create PuLP problem
        prob = LpProblem("EdgeServerPlacement", LpMinimize)

        # Define placement variables
        placement_vars = [LpVariable(f"place_{i}", cat=LpBinary) for i in range(self.n)]

        # Define assigned variables
        assigned_vars = [LpVariable(f"assigned_{i}", cat=LpBinary) for i in range(self.n)]

        # Objective function
        prob += lpSum(self.weights[i] * placement_vars[i] for i in range(self.n))

        # Constraint: Total number of edge servers should be K
        prob += lpSum(placement_vars) == self.k

        # Constraint: Whether a base station has been assigned to an edge server
        for bsid, esids in enumerate(self.belongs):
            prob += lpSum(placement_vars[esid] for esid in esids) >= assigned_vars[bsid]

        # Constraint: The total number of assigned base stations
        acceptable = int(self.n * 0.9)
        prob += lpSum(assigned_vars) >= acceptable

        # Solve the problem
        prob.solve()

        if prob.status == 1:  # Optimal solution found
            print("Solution value =", value(prob.objective))
            places = [i for i, var in enumerate(placement_vars) if value(var) == 1]
            print("Edge servers placed at:", places)
            self.process_result(places)
        else:
            print("No solution available")

        logging.info("{0}:End running MIP".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def preprocess_problem(self):
        base_stations = self.base_stations[:self.n]
        d = np.array([row[:self.n] for row in self.distances[:self.n]])
        cap = int(len(base_stations) / self.k)
        assign = []
        max_distances = []

        for i, row in enumerate(d):
            indices = row.argpartition(cap)[:cap]
            assign.append(indices)
            max_distances.append(row[indices].max())

        avg_workload = sum(bs.workload for bs in base_stations) / self.k
        workload_diff = []

        for row in assign:
            assigned_stations = [base_stations[i] for i in row]
            workload = sum(bs.workload for bs in assigned_stations)
            workload_diff.append((workload - avg_workload) ** 2)

        normalized_max_distances = MIPServerPlacer._normalize(max_distances)
        normalized_workload_diff = MIPServerPlacer._normalize(workload_diff)

        alpha = 0.5
        self.weights = [
            alpha * normalized_max_distances[i] + (1 - alpha) * normalized_workload_diff[i]
            for i in range(self.n)
        ]

        self.belongs = [[] for _ in range(self.n)]
        for i, row in enumerate(assign):
            for bs in row:
                self.belongs[bs].append(i)

        self.assign = assign

    def process_result(self, solution):
        base_stations = self.base_stations[:self.n]
        edge_servers = [
            EdgeServer(i, base_stations[x].latitude, base_stations[x].longitude, base_stations[x].id)
            for i, x in enumerate(solution)
        ]

        for base_station in base_stations:
            closest_edge_server = min(
                edge_servers,
                key=lambda es: self._distance_edge_server_base_station(es, base_station),
            )
            closest_edge_server.assigned_base_stations.append(base_station)
            closest_edge_server.workload += base_station.workload

        self.edge_servers = edge_servers

    @staticmethod
    def _normalize(values: Iterable):
        minimum = min(values)
        delta = max(values) - minimum
        return [(v - minimum) / delta for v in values]