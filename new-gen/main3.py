import time
import pandas as pd
import logging
from datetime import datetime
import numpy as np

from algo.miqp import MIQPServerPlacer
from algo.mip import MIPServerPlacer
from algo.random import RandomServerPlacer
from algo.kmeans import KMeansServerPlacer
from algo.topk import TopKServerPlacer
from algo.qpso import QPSOServerPlacer
from utils import DataUtils

def run_with_settings(placer, n, k):
    # Jalankan penempatan server sekali
    placer.place_server(n, k)
    # Objective (1): rata-rata workload dari semua ES
    avg_workload = np.mean([es.workload for es in placer.edge_servers])
    # Objective (2): rata-rata communication delay antara BS dan ES
    avg_delay = placer.objective_latency()
    return avg_workload, avg_delay

def get_es_locations(placer):
    """
    Menghasilkan representasi string dari lokasi masing-masing ES,
    misalnya: "ES 0: (lat, lng); ES 1: (lat, lng); ..."
    """
    locations = "; ".join([f"ES {es.id}: ({es.latitude:.5f}, {es.longitude:.5f})" for es in placer.edge_servers])
    return locations

def get_bs_assignment(placer):
    """
    Menghasilkan representasi string skema penempatan BS ke ES.
    Contoh: "ES 0: [BS 0, BS 3, BS 7]; ES 1: [BS 1, BS 2, BS 5]; ..."
    """
    assignments = "; ".join(
        [f"ES {es.id}: [{', '.join(str(bs.id) for bs in es.assigned_base_stations)}]" 
         for es in placer.edge_servers]
    )
    return assignments

def run(placers, results_fpath='results/results3.csv'):
    n = 2000
    records = []
    # Variasi jumlah edge server (K)
    for k in range(100, 600, 100):
        print(f'\nSettings: N = {n}, K = {k}')
        for name, placer in placers.items():
            print(f"\nAlgorithm: {name}")
            # Jalankan algoritma penempatan server
            placer.place_server(n, k)
            # Ambil representasi lokasi ES dan skema penempatan BS ke ES
            es_locations = get_es_locations(placer)
            bs_assignment = get_bs_assignment(placer)
            # Hitung objective
            avg_workload = np.mean([es.workload for es in placer.edge_servers])
            avg_delay = placer.objective_latency()
            # Tampilkan informasi ke konsol
            print("ES Locations:")
            print(es_locations)
            print("BS Assignment:")
            print(bs_assignment)
            print(f"Objective (1) - Average Workload: {avg_workload}")
            print(f"Objective (2) - Average Communication Delay: {avg_delay}")
            # Simpan record hasil
            record = {
                'num_base_stations': n,
                'num_edge_servers': k,
                'placer_name': name,
                'ES_locations': es_locations,
                'BS_assignment': bs_assignment,
                'avg_workload': avg_workload,
                'avg_delay': avg_delay
            }
            records.append(record)
    # Simpan semua record ke file CSV
    pd_records = pd.DataFrame(records)
    pd_records.to_csv(results_fpath, index=False)
    print(f"\nHasil telah disimpan ke {results_fpath}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Inisialisasi data dari file dataset
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    # Daftar algoritma penempatan server
    placers = {
        # 'MIQP': MIQPServerPlacer(data.base_stations, data.distances),
        # 'MIP': MIPServerPlacer(data.base_stations, data.distances),
        'K-means': KMeansServerPlacer(data.base_stations, data.distances),
        'Top-K': TopKServerPlacer(data.base_stations, data.distances),
        'Random': RandomServerPlacer(data.base_stations, data.distances),
        'QPSO': QPSOServerPlacer(data.base_stations, data.distances)
    }
    run(placers)
