import pandas as pd
import matplotlib.pyplot as plt

# Baca file CSV hasil
df = pd.read_csv('results/results2.csv')

# Tampilkan beberapa baris data untuk pengecekan
print(df.head())

# Dapatkan daftar algoritma dan jumlah edge server yang unik
algorithms = df['placer_name'].unique()
edge_servers = sorted(df['num_edge_servers'].unique())

# Buat figure dengan dua subplot: satu untuk workload dan satu untuk delay
plt.figure(figsize=(14, 6))

# Subplot 1: Average Workload vs. Jumlah Edge Server
plt.subplot(1, 2, 1)
for alg in algorithms:
    subset = df[df['placer_name'] == alg].sort_values('num_edge_servers')
    plt.plot(subset['num_edge_servers'], subset['avg_workload'], marker='o', label=alg)
plt.xlabel('Jumlah Edge Server')
plt.ylabel('Rata-rata Workload')
plt.title('Perbandingan Rata-rata Workload')
plt.legend()
plt.grid(True)

# Subplot 2: Average Communication Delay vs. Jumlah Edge Server
plt.subplot(1, 2, 2)
for alg in algorithms:
    subset = df[df['placer_name'] == alg].sort_values('num_edge_servers')
    plt.plot(subset['num_edge_servers'], subset['avg_delay'], marker='o', label=alg)
plt.xlabel('Jumlah Edge Server')
plt.ylabel('Rata-rata Communication Delay')
plt.title('Perbandingan Rata-rata Communication Delay')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()