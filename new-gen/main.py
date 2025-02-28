# main.py
import logging
from data_processing import load_base_stations, aggregate_user_data, compute_potential_user, filter_base_stations
from qpso import QPSOServerPlacer
import pandas as pd

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Lokasi dataset (sesuaikan path sesuai struktur direktori Anda)
    bs_csv = 'bs_all.csv'
    user_csv = 'data_all.csv'
    
    # 1. Muat data Base Station
    base_stations = load_base_stations(bs_csv)
    logging.info(f"Loaded {len(base_stations)} base stations.")
    
    # 2. Agregasi data transaksi pengguna
    base_stations = aggregate_user_data(base_stations, user_csv)
    
    # 3. Hitung skor potential user untuk tiap Base Station
    base_stations = compute_potential_user(base_stations)
    
    # 4. Preprocessing: filter BS dengan potential_user di bawah threshold
    threshold = 0.1  # sesuaikan threshold sesuai analisis Anda
    candidate_bs = filter_base_stations(base_stations, threshold)
    logging.info(f"After filtering, {len(candidate_bs)} candidate base stations remain.")
    
    # 5. Parameter QPSO
    K = 5  # jumlah edge server yang akan ditempatkan (sesuaikan)
    swarm_size = 30
    iterations = 50
    beta_qpso = 0.75
    alpha_delay = 0.5
    beta_workload = 0.3
    gamma_potential = 0.2
    distance_threshold = 10  # km; base station yang lebih jauh diberi penalti
    
    qpso_placer = QPSOServerPlacer(candidate_bs, K, swarm_size, iterations, beta_qpso,
                                   alpha_delay, beta_workload, gamma_potential, distance_threshold)
    
    # 6. Jalankan optimasi QPSO
    best_edge_servers, best_obj = qpso_placer.optimize()
    logging.info(f"Optimization completed. Best objective value: {best_obj}")
    
    # Tampilkan hasil edge server dan assignment-nya
    for es in best_edge_servers:
        logging.info(f"Edge Server {es.id} at ({es.latitude}, {es.longitude}), serves {len(es.assigned_base_stations)} base stations, total workload {es.workload}")
    
    # Simpan hasil assignment ke CSV
    results = []
    for es in best_edge_servers:
        for bs in es.assigned_base_stations:
            results.append({
                'edge_server_id': es.id,
                'edge_server_latitude': es.latitude,
                'edge_server_longitude': es.longitude,
                'base_station_id': bs.id,
                'base_station_latitude': bs.latitude,
                'base_station_longitude': bs.longitude,
                'base_station_potential': bs.potential_user,
                'base_station_workload': bs.workload
            })
    df_results = pd.DataFrame(results)
    df_results.to_csv('results.csv', index=False)
    logging.info("Results saved to results.csv")
    
if __name__ == '__main__':
    main()