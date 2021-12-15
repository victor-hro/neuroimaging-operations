def norm_wholebrain(img, noise=False):
    # torch.Size([c, sag, cor, axi])
    voxel = img[img > 0]
    mean = voxel.mean()
    std = voxel.std()

    # torch.Size([c, sag, cor, axi])
    out = torch.Tensor.cuda((img - mean) / std)

    if noise == True:
        out_random = torch.normal(0, 1, size=img.shape)
        out[img == 0] = out_random[img == 0]
    return out
