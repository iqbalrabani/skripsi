import logging
import random
import numpy as np
from datetime import datetime
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

from algo.server_placer import ServerPlacer
from data.edge_server import EdgeServer


class MIQPServerPlacer(ServerPlacer):
    """
    MIQP base heuristic using PuLP
    """
    name = 'MIQP'

    def __init__(self, base_stations, distances):
        super().__init__(base_stations, distances)
        self.n = 0
        self.k = 0
        self.workloads = np.array([bs.workload for bs in base_stations])
        self.avg_workload = None
        self.ln_coefs = None
        self.qmat = None
        self.dvars = None

    def place_server(self, base_station_num, edge_server_num):
        logging.info("{0}: Start running MIQP with N={1}, K={2}".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            base_station_num, edge_server_num))

        self.n = base_station_num
        self.k = edge_server_num
        distances = np.array([self.distances[i][:self.n] for i in range(self.n)])

        self.preprocess()

        locations = [1] * self.k + [0] * (self.n - self.k)
        random.shuffle(locations)

        prob, variables = self.setup_problem(locations)
        prob.solve()

        solutions = np.array(
            [[int(variables['x_{0}_{1}'.format(i, l)].varValue) for l in range(self.n)] for i in range(self.n)]
        )

        while True:
            centers = [0] * self.n
            for l, v in enumerate(locations):
                if v == 1:
                    min_dist = 1e10
                    position = None
                    mask = solutions[:, l]
                    if mask.sum() == 0:
                        logging.warning("Empty edge server!")
                        centers[l] = 1
                        continue

                    for ind, val in enumerate(mask):
                        if val == 1:
                            t = np.sum(distances[ind] * mask)
                            if t < min_dist:
                                min_dist = t
                                position = ind

                    centers[position] = 1

            if centers == locations:
                self.process_result(solutions, locations)
                break

            locations = centers
            prob, variables = self.setup_problem(locations)
            prob.solve()
            solutions = np.array(
                [[int(variables['x_{0}_{1}'.format(i, l)].varValue) for l in range(self.n)] for i in range(self.n)]
            )

        logging.info("{0}: End running MIQP".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def preprocess(self):
        mu = 0.5
        wl = self.workloads[:self.n]
        dist_ln = np.array([self.distances[i][j] for i in range(self.n) for j in range(self.n)])
        wl_ln = np.array([self.workloads[i] for i in range(self.n) for j in range(self.n)])

        avg_workload = np.average(wl)
        wb_max = np.var([np.sum(wl)] + [0] * (self.k - 1))
        dist_max = self.n * np.max(dist_ln)

        ln_coefs = -2 * mu * wl_ln * avg_workload / self.k / wb_max + (1 - mu) * dist_ln / dist_max
        self.ln_coefs = ln_coefs.reshape(self.n, self.n)

    def setup_problem(self, locations):
        prob = LpProblem("MIQP", LpMinimize)

        variables = {
            f"x_{i}_{j}": LpVariable(f"x_{i}_{j}", cat=LpBinary)
            for i in range(self.n) for j in range(self.n)
        }

        # Objective function
        prob += lpSum(self.ln_coefs[i, j] * variables[f"x_{i}_{j}"]
                      for i in range(self.n) for j in range(self.n))

        # Constraints: sum of x_i,j over j equals 1 for each i
        for i in range(self.n):
            prob += lpSum(variables[f"x_{i}_{j}"] for j in range(self.n)) == 1

        # Constraints: x_i,j <= y_j for each i, j
        for l, y in enumerate(locations):
            for i in range(self.n):
                prob += variables[f"x_{i}_{l}"] <= y

        return prob, variables

    def process_result(self, solution, locations):
        base_stations = self.base_stations[:self.n]
        positions = [l for l, i in enumerate(locations) if i == 1]
        edge_servers = [EdgeServer(i, base_stations[x].latitude, base_stations[x].longitude, base_stations[x].id)
                        for i, x in enumerate(positions)]
        for i, p in enumerate(positions):
            for j in range(self.n):
                if solution[j][p] == 1:
                    edge_servers[i].assigned_base_stations.append(base_stations[j])
                    edge_servers[i].workload += base_stations[j].workload
        self.edge_servers = edge_servers