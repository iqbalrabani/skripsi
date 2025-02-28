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

def run_with_settings(placer, n, k, repeat_times=1):
    if repeat_times == 1:
        placer.place_server(n, k)
        # Hitung objective:
        # Objective (1): Rata-rata workload ES
        avg_workload = np.mean([es.workload for es in placer.edge_servers])
        # Objective (2): Rata-rata communication delay antara BS dan ES
        avg_delay = placer.objective_latency()
        objectives = {'avg_workload': avg_workload, 'avg_delay': avg_delay}
    else:
        workload_list = []
        delay_list = []
        for t in range(repeat_times):
            placer.place_server(n, k)
            avg_workload = np.mean([es.workload for es in placer.edge_servers])
            avg_delay = placer.objective_latency()
            workload_list.append(avg_workload)
            delay_list.append(avg_delay)
            time.sleep(1)
        objectives = {'avg_workload': sum(workload_list)/len(workload_list), 'avg_delay': sum(delay_list)/len(delay_list)}
    return objectives

def print_es_locations(placer):
    print("Edge Server Locations:")
    for es in placer.edge_servers:
        print(f"ES {es.id}: (lat: {es.latitude}, lng: {es.longitude}) - Assigned BS: {len(es.assigned_base_stations)}")

def run(placers, results_fpath='results/results.csv'):
    n = 2000
    records = []
    for k in range(100, 600, 100):
        print(f'\nSettings: N={n}, K={k}')
        for name, placer in placers.items():
            print(f"\nAlgorithm: {name}")
            placer.place_server(n, k)
            print_es_locations(placer)
            avg_workload = np.mean([es.workload for es in placer.edge_servers])
            avg_delay = placer.objective_latency()
            print(f"Objective (1) - Average Workload: {avg_workload}")
            print(f"Objective (2) - Average Communication Delay: {avg_delay}")
            settings = {'num_base_stations': n, 'num_edge_servers': k, 'placer_name': name}
            record = {**settings, 'avg_workload': avg_workload, 'avg_delay': avg_delay}
            records.append(record)
    pd_records = pd.DataFrame(records)
    pd_records.to_csv(results_fpath, index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = DataUtils('./dataset/bs_all.csv', './dataset/data_all.csv')
    placers = {
        'MIQP': MIQPServerPlacer(data.base_stations, data.distances),
        'MIP': MIPServerPlacer(data.base_stations, data.distances),
        'K-means': KMeansServerPlacer(data.base_stations, data.distances),
        'Top-K': TopKServerPlacer(data.base_stations, data.distances),
        'Random': RandomServerPlacer(data.base_stations, data.distances),
        'QPSO': QPSOServerPlacer(data.base_stations, data.distances)
    }
    run(placers)