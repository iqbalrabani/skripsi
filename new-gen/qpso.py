# qpso.py
import numpy as np
import random
import math
from base_station import BaseStation
from edge_server import EdgeServer
import logging

def calc_distance(lat_a, lng_a, lat_b, lng_b):
    """
    Menghitung jarak antara dua titik menggunakan formula Haversine.
    Hasil dalam kilometer.
    """
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - math.cos((lat_b - lat_a) * p) / 2 + math.cos(lat_a * p) * math.cos(lat_b * p) * (1 - math.cos((lng_b - lng_a) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))  # 2*R*asin...

class QPSOServerPlacer:
    """
    Kelas untuk optimasi penempatan edge server menggunakan QPSO berdasarkan potential user.
    
    Representasi partikel: vektor kontinu (dimensi N, jumlah kandidat BS).
    Solusi biner diperoleh dengan memilih top K indeks (memastikan tepat K BS terpilih).
    """
    def __init__(self, candidate_bs, K, swarm_size=30, iterations=100, beta=0.75,
                 alpha_delay=0.5, beta_workload=0.3, gamma_potential=0.2, distance_threshold=None):
        """
        Parameters:
            candidate_bs: list of BaseStation (hasil filter preprocessing)
            K: jumlah edge server yang ingin ditempatkan
            swarm_size: jumlah partikel dalam swarm
            iterations: jumlah iterasi QPSO
            beta: parameter QPSO
            alpha_delay, beta_workload, gamma_potential: bobot untuk komponen fungsi tujuan
            distance_threshold: ambang batas jarak (km) untuk assignment; jika tidak diberikan, dianggap tak terbatas
        """
        self.candidate_bs = candidate_bs
        self.N = len(candidate_bs)  # jumlah kandidat BS
        self.K = K  # jumlah edge server
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.beta = beta  # parameter QPSO
        self.alpha_delay = alpha_delay
        self.beta_workload_weight = beta_workload
        self.gamma_potential = gamma_potential
        self.distance_threshold = distance_threshold if distance_threshold is not None else float('inf')
        
        # Precompute matriks jarak antar kandidat BS
        self.distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.distances[i, j] = calc_distance(candidate_bs[i].latitude, candidate_bs[i].longitude,
                                                      candidate_bs[j].latitude, candidate_bs[j].longitude)
    
    def objective_function(self, particle_binary):
        """
        Fungsi tujuan untuk suatu solusi biner (vektor 0/1 dengan tepat K nilai 1).
        Langkah evaluasi:
          1. Tentukan BS yang terpilih sebagai edge server (indeks di mana bit == 1).
          2. Untuk setiap kandidat BS, assign ke edge server terdekat (jika jarak melebihi threshold, beri penalti).
          3. Hitung rata-rata delay, workload masing-masing edge server, dan workload imbalance.
          4. Hitung total potential coverage (jumlah skor potential_user dari BS yang terpilih).
          5. Fungsi tujuan:  
             objective = (alpha_delay * avg_delay) + (beta_workload * workload_imbalance) – (gamma_potential * potential_coverage)
        """
        selected_indices = [i for i, bit in enumerate(particle_binary) if bit == 1]
        if len(selected_indices) == 0:
            return float('inf')
        
        total_delay = 0
        assignments = [-1] * self.N
        workload_dict = {i: 0 for i in selected_indices}
        for i in range(self.N):
            distances = [self.distances[i, j] for j in selected_indices]
            min_distance = min(distances)
            assigned_server = selected_indices[distances.index(min_distance)]
            # Jika jarak melebihi threshold, tambahkan penalti
            if min_distance > self.distance_threshold:
                min_distance *= 10
            total_delay += min_distance
            assignments[i] = assigned_server
            workload_dict[assigned_server] += self.candidate_bs[i].workload
        
        avg_delay = total_delay / self.N
        workloads = list(workload_dict.values())
        workload_imbalance = max(workloads) - min(workloads) if workloads else 0
        potential_coverage = sum([self.candidate_bs[i].potential_user for i in selected_indices])
        
        objective = self.alpha_delay * avg_delay + self.beta_workload_weight * workload_imbalance - self.gamma_potential * potential_coverage
        return objective

    def repair_particle(self, particle_binary):
        """
        Memastikan vektor biner memiliki tepat K angka 1.
        Jika jumlah angka 1 tidak sama dengan K, lakukan perbaikan:
          - Jika lebih, set secara acak sebagian 1 menjadi 0.
          - Jika kurang, tambahkan 1 secara acak sampai tepat K.
        """
        ones = sum(particle_binary)
        if ones > self.K:
            indices = [i for i, bit in enumerate(particle_binary) if bit == 1]
            remove_count = ones - self.K
            remove_indices = random.sample(indices, remove_count)
            for i in remove_indices:
                particle_binary[i] = 0
        elif ones < self.K:
            indices = [i for i, bit in enumerate(particle_binary) if bit == 0]
            add_count = self.K - ones
            add_indices = random.sample(indices, add_count)
            for i in add_indices:
                particle_binary[i] = 1
        return particle_binary

    def continuous_to_binary(self, particle_continuous):
        """
        Mengubah vektor kontinu menjadi solusi biner dengan memilih top K indeks.
        """
        indices = np.argsort(particle_continuous)[-self.K:]  # ambil indeks dengan nilai tertinggi
        binary = [0] * self.N
        for idx in indices:
            binary[idx] = 1
        return binary

    def optimize(self):
        """
        Menjalankan algoritma QPSO dan mengembalikan solusi terbaik yang ditemukan.
        
        Returns:
            best_edge_servers: list of EdgeServer yang dihasilkan.
            best_objective: nilai fungsi tujuan terbaik.
        """
        # Inisialisasi swarm: setiap partikel adalah vektor kontinu (nilai acak antara 0 dan 1)
        swarm = [np.random.rand(self.N) for _ in range(self.swarm_size)]
        pbest_continuous = swarm.copy()
        pbest_binary = [self.continuous_to_binary(p) for p in swarm]
        pbest_obj = [self.objective_function(self.repair_particle(pb.copy())) for pb in pbest_binary]
        
        # Global best
        gbest_index = np.argmin(pbest_obj)
        gbest_continuous = pbest_continuous[gbest_index].copy()
        gbest_binary = pbest_binary[gbest_index].copy()
        gbest_obj = pbest_obj[gbest_index]
        
        for it in range(self.iterations):
            mbest = np.mean(pbest_continuous, axis=0)
            for i in range(self.swarm_size):
                for d in range(self.N):
                    u = random.random()
                    sign = 1 if random.random() < 0.5 else -1
                    pbest_val = pbest_continuous[i][d]
                    mbest_d = mbest[d]
                    if u == 0:
                        u = 1e-10
                    new_val = mbest_d + sign * self.beta * abs(pbest_val - mbest_d) * math.log(1/u)
                    new_val = max(0, min(1, new_val))
                    swarm[i][d] = new_val
                particle_binary = self.continuous_to_binary(swarm[i])
                particle_binary = self.repair_particle(particle_binary)
                obj_val = self.objective_function(particle_binary)
                if obj_val < pbest_obj[i]:
                    pbest_obj[i] = obj_val
                    pbest_continuous[i] = swarm[i].copy()
                    pbest_binary[i] = particle_binary.copy()
                if obj_val < gbest_obj:
                    gbest_obj = obj_val
                    gbest_continuous = swarm[i].copy()
                    gbest_binary = particle_binary.copy()
            logging.info(f"Iteration {it+1}/{self.iterations}, best objective = {gbest_obj}")
        
        # Bangun edge server berdasarkan solusi terbaik
        best_edge_servers = []
        selected_indices = [i for i, bit in enumerate(gbest_binary) if bit == 1]
        for idx, bs_idx in enumerate(selected_indices):
            bs = self.candidate_bs[bs_idx]
            edge_server = EdgeServer(id=idx, latitude=bs.latitude, longitude=bs.longitude, base_station_id=bs.id)
            best_edge_servers.append(edge_server)
        # Lakukan assignment: setiap kandidat BS diassign ke edge server terdekat
        for i, bs in enumerate(self.candidate_bs):
            min_distance = float('inf')
            assigned_server = None
            for es in best_edge_servers:
                dist = calc_distance(bs.latitude, bs.longitude, es.latitude, es.longitude)
                if dist < min_distance:
                    min_distance = dist
                    assigned_server = es
            if assigned_server is not None:
                assigned_server.assigned_base_stations.append(bs)
                assigned_server.workload += bs.workload
        
        return best_edge_servers, gbest_obj