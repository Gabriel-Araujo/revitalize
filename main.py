import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import cv2

from procedures.pre_processing import load_pairs, extract_text_patches_strict
from procedures.model import build_unet_autoencoder

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Callback de diagnóstico
class PredStats(tf.keras.callbacks.Callback):
    def __init__(self, sample_batch):
        super().__init__()
        self.sample = sample_batch
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.sample, verbose=0)
        print(f"[dbg] epoch {epoch} pred min={preds.min():.3f} max={preds.max():.3f} mean={preds.mean():.3f}")

def validate_dataset(clean_dir, degraded_dir, img_size=128, sample=8):
    files = sorted([f for f in os.listdir(clean_dir)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.bmp'))])
    miss = [f for f in files if not os.path.exists(os.path.join(degraded_dir, f))]
    if miss:
        print(f"Faltando {len(miss)} pares. Ex: {miss[:5]}")
    else:
        print("Todos os nomes têm par degradado.")
    if not files: return
    pick = np.random.choice(files, size=min(sample, len(files)), replace=False)
    psnrs, ssims = [], []
    for fname in pick:
        c = cv2.imread(os.path.join(clean_dir, fname), cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(os.path.join(degraded_dir, fname), cv2.IMREAD_GRAYSCALE)
        if c is None or d is None: continue
        c = cv2.resize(c, (img_size, img_size)).astype(np.float32)/255.0
        d = cv2.resize(d, (img_size, img_size)).astype(np.float32)/255.0
        psnrs.append(tf.image.psnr(c[None,...,None], d[None,...,None], max_val=1.0).numpy()[0])
        ssims.append(ssim(c, d, data_range=1.0))
    if psnrs:
        print(f"PSNR base(médio): {np.mean(psnrs):.2f}  SSIM base: {np.mean(ssims):.3f}")

def main():
    # Reprodutibilidade
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    CLEAN_DIR = "dataset/train/clean/letter"
    DEGRADED_DIR = "dataset/train/degraded"

    TARGET_FULL = 1000   # resize + pad
    PATCH       = 320
    SUBSET      = 120    # páginas (reduza/aumente)
    MAX_PATCHES = 12    # patches por página

    validate_dataset(CLEAN_DIR, DEGRADED_DIR, img_size=128)

    # Carrega páginas (limit aplicado antes da leitura na função otimizada)
    X_full, Y_full = load_pairs(CLEAN_DIR,
                                img_size=TARGET_FULL,
                                generate=False,
                                degraded_dir=DEGRADED_DIR,
                                limit=SUBSET,
                                keep_ratio=True,
                                show_progress=True)
    print("Full pages:", X_full.shape)

    # Extrai patches com texto
    X_p, Y_p = extract_text_patches_strict(X_full, Y_full,
                                           patch=PATCH,
                                           max_patches_per_img=MAX_PATCHES)
    if X_p is None:
        print("Nenhum patch gerado. Ajuste parâmetros.")
        return

    # Estatística de texto
    dark_prop = (Y_p < 0.90).mean(axis=(1,2,3))
    print(f"Proporção média de texto: {dark_prop.mean():.4f}  Faixa: {dark_prop.min():.4f} -> {dark_prop.max():.4f}")

    print("Patches:", X_p.shape)

    # Visual rápida de alguns patches limpos
    for i in range(min(3, len(Y_p))):
        plt.figure(figsize=(4,2))
        plt.imshow(Y_p[i].squeeze(), cmap='gray')
        plt.title(f"dark_ratio={ (Y_p[i]<0.9).mean():.3f}")
        plt.axis('off')
        plt.show()

    # Split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_p, Y_p, test_size=0.1, random_state=SEED)

    # Modelo
    model = build_unet_autoencoder(input_shape=(PATCH, PATCH, 1))

    # Callbacks
    callbacks = [
        EarlyStopping(patience=8, monitor="val_loss", restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
        ModelCheckpoint("best_unet.h5", monitor="val_loss", save_best_only=True),
        PredStats(X_val[:2])  # debug
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=80,
        batch_size=6,
        shuffle=True,
        callbacks=callbacks
    )

    # Predições
    preds = model.predict(X_val[:6])
    for i in range(min(3,len(preds))):
        plt.figure(figsize=(8,3))
        plt.subplot(1,3,1); plt.imshow(X_val[i].squeeze(), cmap='gray'); plt.title("Degradada"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(preds[i].squeeze(), cmap='gray'); plt.title("Reconstruída"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(Y_val[i].squeeze(), cmap='gray'); plt.title("Original"); plt.axis('off')
        plt.tight_layout(); plt.show()

    sample_file = os.path.join(DEGRADED_DIR, os.listdir(DEGRADED_DIR)[0])
    from procedures.pre_processing import reconstruct_document, resize_with_pad
    raw_orig, rec_page = reconstruct_document(model, sample_file,
                                              target_full=TARGET_FULL,
                                              patch=PATCH, stride=PATCH//2)

    rec_gray = rec_page

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    rec_clahe = clahe.apply(rec_gray)

    rec_bin = cv2.adaptiveThreshold(
        rec_clahe, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    base_name = os.path.basename(sample_file)
    clean_path = os.path.join(CLEAN_DIR, base_name)
    clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    if clean is not None:
        clean_pad = resize_with_pad(clean, TARGET_FULL)

    plt.figure(figsize=(14,4))
    plt.subplot(1,4,1); plt.imshow(raw_orig, cmap='gray'); plt.title("Degradada"); plt.axis('off')
    plt.subplot(1,4,2); plt.imshow(rec_page, cmap='gray'); plt.title("Reconstruída"); plt.axis('off')
    plt.subplot(1,4,3); plt.imshow(rec_clahe, cmap='gray'); plt.title("CLAHE"); plt.axis('off')
    plt.subplot(1,4,4); plt.imshow(rec_bin, cmap='gray'); plt.title("Binarizada"); plt.axis('off')
    plt.tight_layout(); plt.show()

    # Salvar resultados
    cv2.imwrite("out_reconstruida.png", rec_page)
    cv2.imwrite("out_reconstruida_clahe.png", rec_clahe)
    cv2.imwrite("out_reconstruida_bin.png", rec_bin)


if __name__ == "__main__":
    main()