import os, argparse, cv2, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model

# ====== Coloque aqui as mesmas funções/objetos usados no treino ======
# Se mudou ALPHA / combined_loss / métricas, copie a versão atual.
ALPHA = 20.0
def psnr_metric(y_true, y_pred): return tf.image.psnr(y_true, y_pred, max_val=1.0)
def ssim_metric(y_true, y_pred): return tf.image.ssim(y_true, y_pred, max_val=1.0)

def combined_loss(y_true, y_pred):
    text_mask = tf.cast(y_true < 0.95, tf.float32)
    dil = tf.nn.max_pool2d(text_mask, 3, 1, 'SAME')
    weights = 1.0 + ALPHA * dil
    l1 = tf.reduce_sum(tf.abs(y_true - y_pred) * weights) / tf.reduce_sum(weights)
    ssim_vals = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_term = 1.0 - tf.reduce_mean(ssim_vals)
    return 0.7*l1 + 0.3*ssim_term

# ====== Utilidades ======
def resize_with_pad(img, target):
    h, w = img.shape
    scale = target / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target), 255, dtype=img.dtype)
    y0 = (target - nh)//2; x0 = (target - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def sliding_reconstruct(model, img_degraded, patch=256, stride=128):
    # img_degraded: float32 [H,W,1] (0-1)
    H,W,_ = img_degraded.shape
    out = np.zeros((H,W), np.float32)
    weight = np.zeros((H,W), np.float32)

    def process(y,x):
        p = img_degraded[y:y+patch, x:x+patch, :]
        pred = model.predict(p[None,...], verbose=0)[0,...,0]
        out[y:y+patch, x:x+patch] += pred
        weight[y:y+patch, x:x+patch] += 1

    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            process(y,x)

    # bordas finais se não encaixa exatamente
    if (H - patch) % stride != 0:
        y = H - patch
        for x in range(0, W - patch + 1, stride):
            process(y,x)
    if (W - patch) % stride != 0:
        x = W - patch
        for y in range(0, H - patch + 1, stride):
            process(y,x)

    weight[weight==0] = 1
    rec = (out/weight).clip(0,1)
    return rec[...,None]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Caminho da imagem degradada")
    ap.add_argument("--model", default="best_unet.h5")
    ap.add_argument("--target_full", type=int, default=768)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--simple", action="store_true",
                    help="Usa somente resize (sem patch sliding)")
    args = ap.parse_args()

    # Carrega modelo
    model = load_model(
        args.model,
        custom_objects={"combined_loss": combined_loss,
                        "psnr_metric": psnr_metric,
                        "ssim_metric": ssim_metric},
        compile=False
    )

    # Lê imagem
    raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print("Imagem não encontrada.")
        return

    if args.simple:
        # Modo rápido (apenas redimensiona e passa direto)
        proc = resize_with_pad(raw, args.patch).astype(np.float32)/255.0
        pred = model.predict(proc[None,...,None], verbose=0)[0,...,0]
        rec_page = (pred*255).astype(np.uint8)
    else:
        # Modo completo com mosaico
        padded = resize_with_pad(raw, args.target_full).astype(np.float32)/255.0
        rec = sliding_reconstruct(model, padded[...,None],
                                  patch=args.patch, stride=args.stride)
        rec_page = (rec.squeeze()*255).astype(np.uint8)

    # Pós-processamento opcional
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    rec_clahe = clahe.apply(rec_page)
    rec_bin = cv2.adaptiveThreshold(rec_clahe,255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,31,10)

    base = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(f"{base}_recon.png", rec_page)
    cv2.imwrite(f"{base}_recon_clahe.png", rec_clahe)
    cv2.imwrite(f"{base}_recon_bin.png", rec_bin)
    print("Salvo:", f"{base}_recon.png", f"{base}_recon_clahe.png", f"{base}_recon_bin.png")

if __name__ == "__main__":
    main()