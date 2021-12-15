def read_image(path, is_tensor=False):
    if os.path.basename(path).split(".")[-1] == "nii":

        if is_tensor == True:
            return torch.Tensor(nib.load(path).get_fdata())
        else:
            return nib.load(path).get_fdata()
