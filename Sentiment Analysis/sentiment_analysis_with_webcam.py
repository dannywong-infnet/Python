import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np


model = YOLO("yolov10n.pt")
cap = cv2.VideoCapture(1)


# Armazenar resultados para cada pessoa detectada (simplificado)
info_cache = {}  # key = id do objeto, value = info_text


def analisar_face(recorte):
    try:
        rgb_img = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
        resultado = DeepFace.analyze(
            img_path=np.array(rgb_img),
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )
        if isinstance(resultado, list):
            resultado = resultado[0]
        age = resultado['age']
        gender = resultado['dominant_gender']
        emotion = resultado['dominant_emotion']
        return f"{gender}, {age}y, {emotion}", age, gender, emotion
    except:
        return "Face não detectada", None, None, None


frame_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]

    print(f"\n{'='*60}")
    print(f"Frame {frame_count}")
    print(f"{'='*60}")

    person_count = 0

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label.lower() == "person":
            person_count += 1
            person_crop = frame[y1:y2, x1:x2]

            # Atualizar DeepFace apenas a cada 10 frames
            if i not in info_cache or frame_count % 10 == 0:
                if person_crop.shape[0] > 50 and person_crop.shape[1] > 50:
                    info_text, age, gender, emotion = analisar_face(person_crop)
                    info_cache[i] = (info_text, age, gender, emotion)
                else:
                    info_cache[i] = ("Recorte muito pequeno", None, None, None)

            info_text, age, gender, emotion = info_cache[i]

            # Print no terminal
            print(f"\nPessoa {person_count}:")
            if age is not None:
                print(f"  Gênero: {gender}")
                print(f"  Idade: {age} anos")
                print(f"  Emoção: {emotion}")
            else:
                print(f"  Status: {info_text}")

            # Desenhar na interface gráfica
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, info_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Print objetos não-pessoa no terminal
            print(f"\nObjeto detectado: {label} (confiança: {conf:.2%})")

    if person_count == 0:
        print("\nNenhuma pessoa detectada neste frame.")

    cv2.imshow("YOLO + DeepFace", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break


cap.release()
cv2.destroyAllWindows()
