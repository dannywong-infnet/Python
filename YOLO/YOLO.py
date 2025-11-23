import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1️⃣ Baixar a imagem da internet
url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# 2️⃣ Carregar modelo YOLO pré-treinado (COCO)
model = YOLO("yolo12x.pt") # ou YOLOv11 se tiver instalado

# 3️⃣ Detectar objetos
results = model(img)

# 4️⃣ Mostrar resultado usando matplotlib
for r in results: # percorrer cada resultado
    annotated_img = r.plot() # retorna numpy array com caixas e labels
    plt.imshow(annotated_img)
    plt.axis("off")
    plt.show()