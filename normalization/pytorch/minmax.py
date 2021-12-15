def norm_minmax(img):
    # ((img - min) / (max - min)) * 255
    return (((img - img.min()) / (img.max() - img.min())) * 255.0).type(torch.uint8)
