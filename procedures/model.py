import tensorflow as tf
from tensorflow.keras import layers, models
from .metrics import psnr_metric, ssim_metric, combined_loss  

IMG_SIZE = 128
CHANNELS = 1  # 1 = grayscale, 3 = RGB

def sep_block(x, f):
    x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(f, 3, padding='same', activation='relu')(x)
    return x

def build_unet_autoencoder(input_shape=(256,256,1), base_filters=16, max_filters=128):
    inp = layers.Input(input_shape)
    f1 = base_filters
    c1 = sep_block(inp, f1); p1 = layers.MaxPooling2D()(c1)         
    f2 = min(f1*2, max_filters)
    c2 = sep_block(p1, f2); p2 = layers.MaxPooling2D()(c2)           
    f3 = min(f2*2, max_filters)
    c3 = sep_block(p2, f3); p3 = layers.MaxPooling2D()(c3)           
    f4 = min(f3*2, max_filters)
    b  = sep_block(p3, f4)                                          

    u3 = layers.UpSampling2D()(b);  u3 = layers.Concatenate()([u3, c3]); c4 = sep_block(u3, f3)
    u2 = layers.UpSampling2D()(c4); u2 = layers.Concatenate()([u2, c2]); c5 = sep_block(u2, f2)
    u1 = layers.UpSampling2D()(c5); u1 = layers.Concatenate()([u1, c1]); c6 = sep_block(u1, f1)

    out = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    m = models.Model(inp, out, name="light_unet")
    m.compile(optimizer='adam', loss=combined_loss,
              metrics=[psnr_metric, ssim_metric])
    return m
