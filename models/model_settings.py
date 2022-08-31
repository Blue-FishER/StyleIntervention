CHANNELS = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 512,
    128: 256,
    256: 128,
    512: 64,
    1024: 32,
}

CONFIG = {
    "size": 1024,
    "style_dim": 512,
    "n_mlp": 8,
    "truncation": 0.7,
    "mean_truncation_num": 4096,  # 用于生成平均截断向量的个数
    "model_path": "checkpoint/stylegan2-ffhq-config-f.pt"
}