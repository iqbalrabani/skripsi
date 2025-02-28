import logging
import pandas as pd

from utils import DataUtils

from algo.random import RandomServerPlacer
from algo.topk import TopKServerPlacer
from algo.kmeans import KMeansServerPlacer
from algo.miqp import MIQPServerPlacer
from algo.mip import MIPServerPlacer
from algo.qpso import QPSOServerPlacer

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # Muat dataset menggunakan DataUtils
    # DataUtils akan membaca file base station dan user info, lalu menghitung jarak antar base station
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    base_stations = data.base_stations
    distances = data.distances

    # Parameter untuk optimasi
    base_station_num = len(base_stations)   # Anda bisa menggunakan seluruh base station atau subset (misalnya 2000)
    edge_server_num = 5                     # Contoh: menempatkan 5 edge server

    # Buat dictionary berisi instance dari masing–masing algoritma
    placers = {}

    # Untuk QPSO, metode yang dipakai adalah optimize() (menghasilkan (best_edge_servers, best_obj))
    placers['QPSO'] = QPSOServerPlacer(
        candidate_bs=base_stations,
        K=edge_server_num,
        swarm_size=30,
        iterations=50,
        beta=0.75,
        alpha_delay=0.5,
        beta_workload=0.3,
        gamma_potential=0.2,
        distance_threshold=10
    )
    placers['Random'] = RandomServerPlacer(base_stations, distances)
    placers['TopK'] = TopKServerPlacer(base_stations, distances)
    placers['KMeans'] = KMeansServerPlacer(base_stations, distances)
    placers['MIQP'] = MIQPServerPlacer(base_stations, distances)
    placers['MIP'] = MIPServerPlacer(base_stations, distances)

    results = []
    # Jalankan setiap algoritma
    for name, placer in placers.items():
        logging.info(f"Running algorithm: {name}")

        # Untuk QPSO, gunakan method optimize(); untuk lainnya, gunakan place_server()
        if name == 'QPSO':
            best_edge_servers, best_obj = placer.optimize()
            # Set hasil solusi ke properti edge_servers agar compute_objectives() dapat dipanggil
            placer.edge_servers = best_edge_servers
        else:
            placer.place_server(base_station_num, edge_server_num)

        # Hitung nilai objektif (rata-rata latency dan workload imbalance)
        objectives = placer.compute_objectives()
        objectives['placer'] = name
        results.append(objectives)
        logging.info(f"Algorithm {name} objectives: {objectives}")

    # Simpan hasil perbandingan ke file CSV
    df = pd.DataFrame(results)
    df.to_csv('results_all.csv', index=False)
    logging.info("All algorithm results saved to results_all.csv")

if __name__ == '__main__':
    main()