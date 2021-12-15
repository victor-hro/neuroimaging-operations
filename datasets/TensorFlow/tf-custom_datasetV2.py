# ============================================================================================================ #
# ==============================================   TF DATASET   ============================================== #
# ============================================================================================================ #


class Dataset:
    def __init__(
        self,
        train_files,
        train_labels,
        validation_files,
        validation_labels,
        batch_size=16,
        augmentation=False,
        **kwargs
    ):
        self.train_files = train_files
        self.train_labels = train_labels
        self.validation_files = validation_files
        self.validation_labels = validation_labels
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.kwargs = kwargs

    
    def preprocess(self, img):   
        img[img < 0] = 0
        
        # percentile normalization
        img[img > np.percentile(img[img > 0], 97)] = np.percentile(img[img > 0], 97)
        
        # standardization
        mask = img > 0  # bool
        mean = np.mean(img[mask])  # mean
        std = np.std(img[mask])  # std
        img[mask] = (img[mask] - mean) / std
        
        if self.augmentation == False:
            return img
        
        elif self.augmentation == True:
            # Augmentation
            transform = A.Compose(self.kwargs.values())
            #new_img = np.copy(img)
            #data = {"image":new_img}
            data = {"image":img}
            aug_data = transform(**data)
            aug_img = aug_data["image"]
            return aug_img


    def load_image(self, file, label):
        img = np.load(file.numpy().decode("utf-8"))  # load image
        img = self.preprocess(img)  # standardize image
        
        img = tf.convert_to_tensor(img, np.float32)  # convert to tensor
                                   
        img = tf.expand_dims(img, axis=2)  # adicionando canal 1 = (91,109,1) para 2mm
        img = tf.image.grayscale_to_rgb(img)  # convertendo para 3 canais = (91,109,3) para 2mm
        return img, label

    # Wraps a python function into a TensorFlow op
    def generator_preprocessing(self, file, labels):
        return tf.py_function(self.load_image, [file, labels], [tf.float32, tf.float32])

    def get_generator(self):

        # --------------------------------------    TRAIN LOADERS    -------------------------------------- #

        train_loader = tf.data.Dataset.from_tensor_slices(
            (train_files, train_labels)
        )
        validation_loader = tf.data.Dataset.from_tensor_slices(
            (validation_files, validation_labels)
        )

        # -----------------------------------   TRAIN AND VAL DATASETS   ----------------------------------- #

        train_dataset = (
            train_loader.shuffle(len(train_files))
            .map(self.generator_preprocessing)
            .batch(self.batch_size)
            .prefetch(self.batch_size)
        )

        val_dataset = (
            validation_loader.shuffle(len(validation_files))
            .map(self.generator_preprocessing)
            .batch(self.batch_size)
            .prefetch(self.batch_size)
        )
        return train_dataset, val_dataset