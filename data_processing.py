# data_processing.py
import pandas as pd
import numpy as np
from base_station import BaseStation
import logging

def load_base_stations(bs_csv_path):
    """
    Memuat data BS dari CSV dan menghasilkan daftar BaseStation.
    """
    bs_data = pd.read_csv(bs_csv_path, index_col=0)
    base_stations = []
    for idx, row in bs_data.iterrows():
        bs = BaseStation(id=int(row['id']), addr=row['address'], lat=float(row['latitude']), lng=float(row['longitude']))
        base_stations.append(bs)
    return base_stations

def aggregate_user_data(base_stations, user_csv_path):
    """
    Menggabungkan data transaksi pengguna ke tiap BaseStation.
    """
    # Mapping dari address ke objek BaseStation
    addr_to_bs = {bs.address: bs for bs in base_stations}
    
    # Memuat data transaksi pengguna
    user_data = pd.read_csv(user_csv_path, index_col=0)
    user_data['start time'] = pd.to_datetime(user_data['start time'])
    user_data['end time'] = pd.to_datetime(user_data['end time'])
    
    # Hitung durasi layanan dalam menit
    user_data['service_time'] = (user_data['end time'] - user_data['start time']).dt.total_seconds() / 60.0
    
    # Agregasi berdasarkan address: total service_time, jumlah pengguna unik, dan jumlah transaksi
    grouped = user_data.groupby('address').agg({
        'service_time': 'sum',
        'user id': pd.Series.nunique,
        'start time': 'count'
    }).rename(columns={'user id': 'unique_users', 'start time': 'transaction_count'})
    
    # Masukkan data agregasi ke masing-masing BaseStation
    for bs in base_stations:
        if bs.address in grouped.index:
            row = grouped.loc[bs.address]
            bs.workload = row['service_time']
            bs.num_users = row['unique_users']
        else:
            bs.workload = 0
            bs.num_users = 0
    return base_stations

def compute_potential_user(base_stations):
    """
    Menghitung skor potential user untuk tiap BaseStation dengan menggabungkan
    jumlah pengguna unik dan total service_time (dengan normalisasi sederhana).
    """
    user_counts = np.array([bs.num_users for bs in base_stations])
    workloads = np.array([bs.workload for bs in base_stations])
    
    # Normalisasi antara 0 dan 1
    if user_counts.max() > user_counts.min():
        norm_users = (user_counts - user_counts.min()) / (user_counts.max() - user_counts.min())
    else:
        norm_users = user_counts
    
    if workloads.max() > workloads.min():
        norm_workloads = (workloads - workloads.min()) / (workloads.max() - workloads.min())
    else:
        norm_workloads = workloads
    
    # Kombinasi sederhana dengan bobot yang sama
    for i, bs in enumerate(base_stations):
        bs.potential_user = 0.5 * norm_users[i] + 0.5 * norm_workloads[i]
    return base_stations

def filter_base_stations(base_stations, threshold=0.1):
    """
    Menyaring BaseStation dengan skor potential_user di bawah threshold.
    """
    filtered = [bs for bs in base_stations if bs.potential_user >= threshold]
    logging.info(f"Filtered out {len(base_stations) - len(filtered)} base stations below threshold {threshold}")
    return filtered