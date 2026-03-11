from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Muat Model
MODEL_PATH = "model/best_old_v2.pt" 
model = YOLO(MODEL_PATH)

kelas_yang_diizinkan = []
for id_kelas, nama_kelas in model.names.items():
    if nama_kelas != "sampah-detection":
        kelas_yang_diizinkan.append(id_kelas)

@app.post("/deteksi/")
async def deteksi_sampah(
    file: UploadFile = File(...), 
    x_api_key: str = Header(None)
):
    # Verifikasi Keamanan
    KUNCI_RAHASIA = "yolo11-deteksi-sampah"
    if x_api_key != KUNCI_RAHASIA:
        raise HTTPException(status_code=401, detail="API Key tidak valid!")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # deteksi
        results = model.predict(source=img, conf=0.48, classes=kelas_yang_diizinkan)

        # menghitung jumlah kotak deteksi pada gambar
        jumlah_deteksi = len(results[0].boxes)

        # gambar kotak di memori
        img_dengan_kotak = results[0].plot()

        # ubah kembali jadi format .JPG
        _, buffer = cv2.imencode('.jpg', img_dengan_kotak)

        # mengubah gambar jadi teks base64 agar bisa masuk JSON
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # kirim ke react sebagai respon gambar
        return JSONResponse(content={
            "jumlah_deteksi": jumlah_deteksi,
            "gambar_base64": img_base64
        })

    except Exception as e:
        print(f"Error Deteksi: {str(e)}")
        raise HTTPException(status_code=500, detail="Gagal memproses gambar pada server.")