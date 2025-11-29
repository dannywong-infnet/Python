import requests
from deepface import DeepFace
from PIL import Image
from io import BytesIO


def analisar_imagem(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content))
    img.show()

    # Análise de atributos faciais
    resultado = DeepFace.analyze(img_path=BytesIO(response.content), actions=['age', 'gender', 'emotion', 'race'])
    print("Resultado:")
    print(f"Idade: {resultado[0]['age']}")
    print(f"Gênero: {resultado[0]['dominant_gender']}")
    print(f"Emoção: {resultado[0]['dominant_emotion']}")
    print(f"Etnia: {resultado[0]['dominant_race']}")

# Exemplo de uso
url_imagem = "https://br.web.img3.acsta.net/c_310_420/pictures/18/08/08/18/29/0662098.jpg"
analisar_imagem(url_imagem)

