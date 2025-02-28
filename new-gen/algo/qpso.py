import logging
import random
import numpy as np
from datetime import datetime
from algo.server_placer import ServerPlacer
from data.edge_server import EdgeServer

class QPSOServerPlacer(ServerPlacer):
    """
    Pendekatan QPSO berbasis potential user.
    Potential user diukur dengan skor komposit:
      composite_score = 0.5 * (normalized workload) + 0.5 * (normalized jumlah pengguna)
    """
    name = "QPSO"

    def place_server(self, base_station_num, edge_server_num):
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Start running QPSO with N={base_station_num}, K={edge_server_num}")
        base_stations = self.base_stations[:base_station_num]
        n = len(base_stations)
        k = edge_server_num

        # Hitung skor komposit untuk masing-masing base station
        composite_scores = self._compute_composite_scores(base_stations)

        # Parameter QPSO
        swarm_size = 20
        max_iter = 50
        mutation_rate = 0.1

        # Inisialisasi swarm: tiap partikel adalah list k indeks unik dari 0 sampai n-1
        swarm = []
        for _ in range(swarm_size):
            particle = sorted(random.sample(range(n), k))
            swarm.append(particle)

        # Fungsi fitness: jumlah skor komposit pada indeks-indeks yang dipilih
        def fitness(particle):
            return sum(composite_scores[i] for i in particle)

        # Inisialisasi pbest (solusi terbaik pribadi) dan gbest (solusi terbaik global)
        pbest = list(swarm)
        pbest_fitness = [fitness(p) for p in swarm]
        gbest = max(swarm, key=fitness)
        gbest_fitness = fitness(gbest)

        # Iterasi QPSO
        for it in range(max_iter):
            for i, particle in enumerate(swarm):
                # Kombinasikan solusi saat ini, pbest dan gbest
                candidate_set = set(particle) | set(pbest[i]) | set(gbest)
                # Jika jumlah kandidat lebih dari k, pilih k BS dengan skor tertinggi
                candidate_list = sorted(candidate_set, key=lambda idx: composite_scores[idx], reverse=True)
                candidate = sorted(candidate_list[:k])

                # Lakukan mutasi: dengan probabilitas mutation_rate, tukar satu elemen secara acak
                if random.random() < mutation_rate:
                    candidate = candidate.copy()
                    remove_idx = random.choice(candidate)
                    candidate.remove(remove_idx)
                    available = set(range(n)) - set(candidate)
                    if available:
                        candidate.append(random.choice(list(available)))
                    candidate = sorted(candidate)

                candidate_fitness = fitness(candidate)
                if candidate_fitness > pbest_fitness[i]:
                    pbest[i] = candidate
                    pbest_fitness[i] = candidate_fitness
                if candidate_fitness > gbest_fitness:
                    gbest = candidate
                    gbest_fitness = candidate_fitness
                swarm[i] = candidate

        # gbest berisi indeks-indeks BS yang dipilih sebagai lokasi ES
        selected_bs = [base_stations[idx] for idx in gbest]
        edge_servers = [EdgeServer(i, bs.latitude, bs.longitude, bs.id) for i, bs in enumerate(selected_bs)]

        # Penugasan setiap base station ke ES terdekat
        for bs in base_stations:
            closest_edge_server = None
            min_distance = float('inf')
            for es in edge_servers:
                d = self._distance_edge_server_base_station(es, bs)
                if d < min_distance:
                    min_distance = d
                    closest_edge_server = es
            closest_edge_server.assigned_base_stations.append(bs)
            closest_edge_server.workload += bs.workload

        self.edge_servers = edge_servers
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: End running QPSO")

    def _compute_composite_scores(self, base_stations):
        # Ambil nilai workload dan jumlah pengguna dari tiap BS
        workloads = [bs.workload for bs in base_stations]
        num_users = [bs.num_users for bs in base_stations]

        norm_workloads = self._normalize(workloads)
        norm_users = self._normalize(num_users)

        # Kombinasi metrik: 50% workload, 50% jumlah pengguna
        composite_scores = [0.5 * w + 0.5 * u for w, u in zip(norm_workloads, norm_users)]
        return composite_scores

    def _normalize(self, values):
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val == 0:
            return [1.0 for _ in values]
        return [(v - min_val) / (max_val - min_val) for v in values]