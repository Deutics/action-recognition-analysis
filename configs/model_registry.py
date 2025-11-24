MODEL_REGISTRY = {
    # pytorchvideo models → all use same loader so thats why using same module
    "x3d_s": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
    "x3d_m": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
    "x3d_l": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
    "r2plus1d_r50": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
    "i3d_r50": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
    "c2d_r50": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
    "csn_r101": {
        "module": "loaders.pytorchvideo_loader.pytorchvideo_loader",
        "class": "PyTorchVideoLoader"
    },
}
