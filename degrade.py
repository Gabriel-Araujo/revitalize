import cv2
import numpy as np
import random
import os

def degrade_image(img):
    # ruido gaussiano
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    degraded = cv2.add(img, noise.astype(np.uint8))

    # manchas aleatórias
    for _ in range(3):  # número de manchas
        x, y = random.randint(0, degraded.shape[1]-1), random.randint(0, degraded.shape[0]-1)
        r = random.randint(10, 50)
        color = random.randint(100, 200)
        cv2.circle(degraded, (x, y), r, (color,), -1)

    # desfoque, gaussian blur 3x3 
    degraded = cv2.GaussianBlur(degraded, (3, 3), 0)

    return degraded

# Diretórios
input_dir = "dataset/train/clean/letter"
output_dir = "dataset/train/degraded"

os.makedirs(output_dir, exist_ok=True)

# Processa todas as imagens da pasta
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Erro ao abrir {filename}")
            continue

        degraded_img = degrade_image(img)

        # Salva versão degradada com o mesmo nome
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, degraded_img)

        print(f"✅ {filename} degradada salva em {save_path}")