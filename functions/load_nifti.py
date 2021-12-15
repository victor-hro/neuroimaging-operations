def load_image_(filename: str):
    # load nifti and transforms into numpy array
    filename = filename.numpy().decode("utf-8")
    # print(filename)
    img = nib.load(filename).get_fdata()
    return img