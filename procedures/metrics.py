import tensorflow as tf

USE_EDGE = False
ALPHA = 20.0  # peso forte para texto

def psnr_metric(y_true, y_pred):  # mede o quão próxima a imagem reconstruída está da original.
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred): # mede a similaridade estrutural entre as imagens. 
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def _edge_map(x):
    gx = tf.image.sobel_edges(x)
    dx = gx[...,0,0]; dy = gx[...,0,1]
    return tf.sqrt(dx*dx + dy*dy + 1e-6)

def combined_loss(y_true, y_pred):
    # Texto = pixels escuros
    text_mask = tf.cast(y_true < 0.95, tf.float32)
    # Dilata texto para cobrir contornos finos
    dil = tf.nn.max_pool2d(text_mask, ksize=3, strides=1, padding='SAME')
    weights = 1.0 + ALPHA * dil  # fundo ~1, texto ~1+ALPHA

    l1 = tf.reduce_sum(tf.abs(y_true - y_pred) * weights) / tf.reduce_sum(weights)
    ssim_vals = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_term = 1.0 - tf.reduce_mean(ssim_vals)

    loss = 0.7 * l1 + 0.3 * ssim_term

    if USE_EDGE:
        e_true = _edge_map(y_true)
        e_pred = _edge_map(y_pred)
        edge_l1 = tf.reduce_mean(tf.abs(e_true - e_pred))
        loss += 0.05 * edge_l1
    return loss