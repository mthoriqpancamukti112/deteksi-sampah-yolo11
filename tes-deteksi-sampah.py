from ultralytics import YOLO

print("=== Memulai Pengujian Banyak Gambar ===")

path_model = "model/best.pt"

# 1. Memuat model
model = YOLO(path_model)

# 2. Tentukan lokasi FOLDER (bukan file spesifik)
# Pastikan folder ini berisi sekumpulan gambar yang ingin Anda tes
path_folder = "media_tes/"

# 3. Lakukan deteksi massal
print(f"Mendeteksi semua file di dalam folder: {path_folder}...")
results = model.predict(
    source=path_folder, 
    show=False,  # Diubah ke False agar layar Anda tidak dibanjiri puluhan jendela popup!
    save=True,   # Menyimpan semua gambar hasil deteksi ke folder runs/detect/predict
    conf=0.4    # Hanya tampilkan kotak jika model yakin minimal 45%
)

print("\n=== Selesai! ===")
print("Silakan cek folder 'runs/detect/predict' untuk melihat semua gambar yang sudah diberi kotak.")