from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from image_processor import auto_enhance
from inpainting import repair_image

from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
import io
from database import get_connection
from io import BytesIO


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    enhanced = auto_enhance(image)
    repaired = repair_image(enhanced)

    _, buffer = cv2.imencode(".jpg", repaired)
    return buffer.tobytes()

# -------------------------
# Endpoint: subir imagen
# -------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Leer imagen original
    original_bytes = await file.read()

    # 👉 Recrear UploadFile en memoria
    memory_file = UploadFile(
        filename=file.filename,
        file=BytesIO(original_bytes)
    )

    # 👉 Llamar exactamente al mismo método
    processed_bytes = await process_image(memory_file)

    # 👉 Guardar en BD
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO ImageRecords (FileName, OriginalImage, ProcessedImage, created_at)
        VALUES (?, ?, ?, GETDATE())
    """

    cursor.execute(sql, file.filename, original_bytes, processed_bytes)
    conn.commit()
    conn.close()

    return {"message": "Imagen procesada usando /process-image y guardada correctamente"}
# 📥 Listar imágenes
@app.get("/images")
def get_images():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT Id, FileName, created_at
        FROM ImageRecords
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row.Id,
            "fileName": row.FileName,
            "created_at": row.created_at
        }
        for row in rows
    ]