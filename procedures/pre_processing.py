from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image, ImageFilter
import random, cv2, os

input_dir = "dataset/train/clean/letter"
output_dir = "dataset/train/degraded"

os.makedirs(output_dir, exist_ok=True)


def resize_with_pad(img, target=256):
    h, w = img.shape
    scale = target / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target), 255, dtype=img.dtype)
    y0 = (target - new_h)//2
    x0 = (target - new_w)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def load_pairs(clean_dir, img_size=256, generate=False, degraded_dir='dataset/train/degraded',
               limit=None, seed=42, keep_ratio=True, show_progress=True):
    rng = np.random.default_rng(seed)
    files = [f for f in os.listdir(clean_dir)
             if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.bmp'))]

    # Aplica limite ANTES de ler (acelera)
    if limit and len(files) > limit:
        files = list(rng.choice(files, size=limit, replace=False))

    files.sort()
    X, Y = [], []
    total = len(files)
    for i, fname in enumerate(files, 1):
        cpath = os.path.join(clean_dir, fname)
        dpath = os.path.join(degraded_dir, fname)
        clean = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE)
        degraded = cv2.imread(dpath, cv2.IMREAD_GRAYSCALE)
        if clean is None or degraded is None:
            continue
        if keep_ratio:
            clean = resize_with_pad(clean, img_size)
            degraded = resize_with_pad(degraded, img_size)
        else:
            clean = cv2.resize(clean, (img_size, img_size))
            degraded = cv2.resize(degraded, (img_size, img_size))
        clean_f = clean.astype(np.float32)/255.0
        degraded_f = degraded.astype(np.float32)/255.0
        X.append(degraded_f[..., None])
        Y.append(clean_f[..., None])
        if show_progress and (i % 50 == 0 or i == total):
            print(f"load_pairs: {i}/{total}")
    return np.array(X), np.array(Y)

def extract_text_patches_strict(X, Y, patch=256, max_patches_per_img=12,
                                var_min=0.0008, dark_min=0.01, dark_max=0.40,
                                attempts=120, thresh_val=0.90):
    """
    Retorna apenas patches com quantidade moderada de texto.
    dark_min: fração mínima de pixels escuros (texto)
    dark_max: fração máxima (evita blocos quase pretos)
    """
    newX, newY = [], []
    for xd, yc in zip(X, Y):
        h, w = xd.shape[:2]
        if h < patch or w < patch:
            continue

        clean = yc[...,0]
        # Texto preto -> valores baixos; máscara texto=1 onde clean < thresh_val
        text_mask = (clean < thresh_val).astype('float32')

        got = 0
        for _ in range(attempts):
            if got >= max_patches_per_img:
                break
            y0 = np.random.randint(0, h - patch + 1)
            x0 = np.random.randint(0, w - patch + 1)
            pd = xd[y0:y0+patch, x0:x0+patch, :]
            if pd.var() < var_min:
                continue
            tm = text_mask[y0:y0+patch, x0:x0+patch]
            dark_ratio = tm.mean()
            if dark_ratio < dark_min or dark_ratio > dark_max:
                continue
            pc = yc[y0:y0+patch, x0:x0+patch, :]
            newX.append(pd); newY.append(pc); got += 1

        if got == 0:
            # fallback central
            cy = (h - patch)//2; cx = (w - patch)//2
            newX.append(xd[cy:cy+patch, cx:cx+patch, :])
            newY.append(yc[cy:cy+patch, cx:cx+patch, :])

    if not newX:
        print("Aviso: nenhum patch gerado. Ajuste parâmetros.")
        return None, None
    return np.stack(newX), np.stack(newY)

def sliding_reconstruct(model, img_degraded, patch=256, stride=192):
    """
    img_degraded: ndarray float32 [H,W,1] já normalizada (0-1) e quadrada (ex: 768x768).
    Retorna imagem reconstruída (0-1) mesma dimensão.
    """
    H, W, _ = img_degraded.shape
    out = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            patch_in = img_degraded[y:y+patch, x:x+patch, :]
            pred = model.predict(patch_in[None,...], verbose=0)[0,...,0]
            out[y:y+patch, x:x+patch] += pred
            weight[y:y+patch, x:x+patch] += 1.0

    # Bordas finais (se não encaixar exato)
    if (H - patch) % stride != 0:
        y = H - patch
        for x in range(0, W - patch + 1, stride):
            patch_in = img_degraded[y:y+patch, x:x+patch, :]
            pred = model.predict(patch_in[None,...], verbose=0)[0,...,0]
            out[y:y+patch, x:x+patch] += pred
            weight[y:y+patch, x:x+patch] += 1.0
    if (W - patch) % stride != 0:
        x = W - patch
        for y in range(0, H - patch + 1, stride):
            patch_in = img_degraded[y:y+patch, x:x+patch, :]
            pred = model.predict(patch_in[None,...], verbose=0)[0,...,0]
            out[y:y+patch, x:x+patch] += pred
            weight[y:y+patch, x:x+patch] += 1.0

    weight[weight==0] = 1
    rec = out / weight
    rec = np.clip(rec, 0, 1)
    return rec[...,None]

def reconstruct_document(model, img_path, target_full=768, patch=256, stride=192):
    """
    Lê imagem degradada (grayscale) do disco, aplica resize_with_pad, reconstrói.
    """
    raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    proc = resize_with_pad(raw, target_full).astype(np.float32)/255.0
    proc = proc[...,None]
    rec = sliding_reconstruct(model, proc, patch=patch, stride=stride)
    return raw, (rec.squeeze()*255).astype(np.uint8)