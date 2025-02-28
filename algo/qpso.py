# qpso.py
import random
import math
import numpy as np
import logging

from datetime import datetime
from typing import List

from algo.server_placer import ServerPlacer
from data.edge_server import EdgeServer
from utils import DataUtils

class QPSOServerPlacer(ServerPlacer):
    """
    QPSO approach for edge server placement based on potential user.
    Representasi partikel: vektor kontinu (dimensi = jumlah candidate base stations).
    Solusi biner diperoleh dengan memilih tepat K kandidat (berdasarkan nilai tertinggi).
    """
    name = 'QPSO'
    
    def __init__(self, base_stations: List, distances: List[List[float]],
                 swarm_size=30, iterations=50, beta=0.75,
                 alpha_delay=0.5, beta_workload=0.3, gamma_potential=0.2,
                 distance_threshold=10):
        super().__init__(base_stations, distances)
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.beta = beta
        self.alpha_delay = alpha_delay
        self.beta_workload = beta_workload
        self.gamma_potential = gamma_potential
        self.distance_threshold = distance_threshold
        self.k = None  # jumlah edge server yang akan dipilih (di-assign pada place_server)
        self.N = len(self.base_stations)
        
        # Precompute matriks jarak antar candidate base stations menggunakan fungsi dari utils
        self.distance_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.distance_matrix[i, j] = DataUtils.calc_distance(
                    self.base_stations[i].latitude, self.base_stations[i].longitude,
                    self.base_stations[j].latitude, self.base_stations[j].longitude)
    
    def objective_function(self, particle_binary: List[int]) -> float:
        """
        Menghitung nilai fungsi tujuan berdasarkan:
          - Average delay (rata-rata jarak antara setiap BS dengan server terdekat)
          - Workload imbalance antar edge server
          - Potential user coverage (nilai potential BS yang terpilih)
        Formula:
            obj = alpha_delay * avg_delay + beta_workload * workload_imbalance - gamma_potential * potential_coverage
        """
        selected_indices = [i for i, bit in enumerate(particle_binary) if bit == 1]
        if len(selected_indices) == 0:
            return float('inf')
        
        total_delay = 0
        workload_dict = {i: 0 for i in selected_indices}
        for i in range(self.N):
            # Hitung jarak ke setiap server candidate yang terpilih
            dists = [self.distance_matrix[i, j] for j in selected_indices]
            min_distance = min(dists)
            if min_distance > self.distance_threshold:
                min_distance *= 10  # Penalti jika melebihi threshold
            total_delay += min_distance
            assigned_idx = selected_indices[dists.index(min_distance)]
            workload_dict[assigned_idx] += self.base_stations[i].workload
        
        avg_delay = total_delay / self.N
        workloads = list(workload_dict.values())
        workload_imbalance = max(workloads) - min(workloads) if workloads else 0
        potential_coverage = sum([self.base_stations[i].potential_user for i in selected_indices])
        
        obj = (self.alpha_delay * avg_delay +
               self.beta_workload * workload_imbalance -
               self.gamma_potential * potential_coverage)
        return obj

    def repair_particle(self, particle_binary: List[int]) -> List[int]:
        """
        Memastikan bahwa solusi biner memiliki tepat self.k angka 1.
        Jika lebih, secara acak diubah menjadi 0; jika kurang, secara acak ditambahkan 1.
        """
        ones = sum(particle_binary)
        if ones > self.k:
            indices = [i for i, bit in enumerate(particle_binary) if bit == 1]
            remove_count = ones - self.k
            remove_indices = random.sample(indices, remove_count)
            for idx in remove_indices:
                particle_binary[idx] = 0
        elif ones < self.k:
            indices = [i for i, bit in enumerate(particle_binary) if bit == 0]
            add_count = self.k - ones
            add_indices = random.sample(indices, add_count)
            for idx in add_indices:
                particle_binary[idx] = 1
        return particle_binary

    def continuous_to_binary(self, particle_continuous: np.ndarray) -> List[int]:
        """
        Mengubah vektor kontinu menjadi solusi biner dengan memilih self.k indeks dengan nilai tertinggi.
        """
        indices = np.argsort(particle_continuous)[-self.k:]
        binary = [0] * self.N
        for idx in indices:
            binary[idx] = 1
        return binary

    def place_server(self, base_station_num, edge_server_num):
        """
        Menjalankan algoritma QPSO untuk memilih candidate base stations sebagai lokasi edge server.
        Setelah solusi ditemukan, base stations diassign ke edge server terdekat.
        """
        logging.info("{0}: Start running QPSO with N={1}, K={2}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                          base_station_num, edge_server_num))
        # Gunakan subset base stations sesuai parameter
        self.k = edge_server_num
        self.N = base_station_num
        self.base_stations = self.base_stations[:base_station_num]
        self.distance_matrix = self.distance_matrix[:base_station_num, :base_station_num]
        
        # Inisialisasi swarm: setiap partikel adalah vektor kontinu dengan nilai acak [0,1]
        swarm = [np.random.rand(self.N) for _ in range(self.swarm_size)]
        pbest_cont = [p.copy() for p in swarm]
        pbest_binary = [self.continuous_to_binary(p.copy()) for p in swarm]
        pbest_obj = [self.objective_function(self.repair_particle(pb.copy())) for pb in pbest_binary]
        
        gbest_index = np.argmin(pbest_obj)
        gbest_cont = pbest_cont[gbest_index].copy()
        gbest_binary = pbest_binary[gbest_index].copy()
        gbest_obj = pbest_obj[gbest_index]
        
        # Iterasi QPSO
        for it in range(self.iterations):
            mbest = np.mean(pbest_cont, axis=0)
            for i in range(self.swarm_size):
                for d in range(self.N):
                    u = random.random() or 1e-10
                    sign = 1 if random.random() < 0.5 else -1
                    new_val = mbest[d] + sign * self.beta * abs(pbest_cont[i][d] - mbest[d]) * math.log(1/u)
                    swarm[i][d] = max(0, min(1, new_val))
                particle_binary = self.continuous_to_binary(swarm[i])
                particle_binary = self.repair_particle(particle_binary)
                obj_val = self.objective_function(particle_binary)
                if obj_val < pbest_obj[i]:
                    pbest_obj[i] = obj_val
                    pbest_cont[i] = swarm[i].copy()
                    pbest_binary[i] = particle_binary.copy()
                if obj_val < gbest_obj:
                    gbest_obj = obj_val
                    gbest_cont = swarm[i].copy()
                    gbest_binary = particle_binary.copy()
            logging.info("{0}: QPSO Iteration {1}/{2}, best objective = {3}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), it+1, self.iterations, gbest_obj))
        
        # Bangun edge server dari kandidat yang terpilih
        selected_indices = [i for i, bit in enumerate(gbest_binary) if bit == 1]
        edge_servers = []
        for idx, bs_idx in enumerate(selected_indices):
            bs = self.base_stations[bs_idx]
            es = EdgeServer(idx, bs.latitude, bs.longitude, bs.id)
            edge_servers.append(es)
        
        # Assignment: setiap base station diassign ke edge server terdekat
        for bs in self.base_stations:
            min_distance = float('inf')
            assigned_es = None
            for es in edge_servers:
                d = DataUtils.calc_distance(bs.latitude, bs.longitude, es.latitude, es.longitude)
                if d < min_distance:
                    min_distance = d
                    assigned_es = es
            if assigned_es is not None:
                assigned_es.assigned_base_stations.append(bs)
                assigned_es.workload += bs.workload
        
        self.edge_servers = edge_servers
        logging.info("{0}: End running QPSO".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))