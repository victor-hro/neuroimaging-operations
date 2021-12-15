import torch
import numpy a numpy
import nibabel as nib

class MyDataset(torch.utils.data.Dataset):
    """
    Dataset class for read nifti files
    reference: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, files, labels, transform=None):
        """
        Parameters:
        -----------
            files [str]: list of paths

            labels [float]: list of labels

            augmentation [dict]
        """
        self.labels = labels
        self.files = files
        self.transform = transform

        if len(self.files) != len(self.labels) or len(self.files) == 0:
            raise ValueError(
                f"Number of source and target images must be equal and non-zero"
            )

    def __len__(self):
        # Denotes the total number of samples"
        return len(self.files)

    def __getitem__(self, index: int):
        # Generates one sample of data

        # Select sample
        img, label = self.files[index], self.labels[index]

        if os.path.basename(img).split(".")[-1] == "nii":
            img = nib.load(img).get_fdata(dtype=np.float32)
            img = np.stack((img,) * 3)  # convert to rgb - out(channels, h, w)
        else:
            img = np.load(img).astype(np.float32)
            img = np.stack((img,) * 3)  # convert to rgb - out(channels, h, w)

        # transform
        if self.transform:
            """All pre-trained models expect input images normalized in the same way,
            i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
            where H and W are expected to be at least 224.
            The images have to be loaded in to a range of [0, 1]
            and then normalized using mean = [0.485, 0.456, 0.406]
            and std = [0.229, 0.224, 0.225]."""

            # move axis - out(h, w, channels)
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) / (img.max() - img.min())
            #             img =  np.expand_dims(img, axis=-1)

            # img = torch.Size([channels, w, h])
            img, label = self.transform(img), torch.Tensor(np.asarray(label))

        return img, label