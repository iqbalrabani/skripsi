# base_station.py
class BaseStation:
    """
    Merepresentasikan sebuah Base Station.
    
    Attributes:
        id: identifier BS
        address: alamat BS (string)
        latitude: koordinat lintang
        longitude: koordinat bujur
        num_users: jumlah unik pengguna (agregasi dari data transaksi)
        workload: total waktu layanan (menit)
        potential_user: skor potensi pengguna (hasil kombinasi num_users dan workload)
    """
    def __init__(self, id, addr, lat, lng):
        self.id = id
        self.address = addr
        self.latitude = lat
        self.longitude = lng
        self.num_users = 0
        self.workload = 0
        self.potential_user = 0  # nilai skor potential user

    def __str__(self):
        return f"No.{self.id}: {self.address}"
