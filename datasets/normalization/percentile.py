def norm_percentile(img, percentile=97):
    img[img < 0] = 0
    img[img > np.percentile(img[img > 0], percentile)] = np.percentile(img[img > 0], percentile)
    return img