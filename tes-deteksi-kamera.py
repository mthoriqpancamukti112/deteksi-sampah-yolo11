from ultralytics import YOLO

print("Memuat model kecerdasan...")
model = YOLO("model/best_old_v2.pt")

print("Model siap! Memulai deteksi...")

model.predict(source=0, show=True, conf=0.48)
print("Deteksi selesai.")