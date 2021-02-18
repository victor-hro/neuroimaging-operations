# ======================================================================================================== #
# ===============================================   DATASETS   =========================================== #
# ======================================================================================================== #

# -------------------------------------------------------------------------------------------------------- #
#                                            LOAD PY FUNCTION 
# -------------------------------------------------------------------------------------------------------- #

def get_generator(batch_size):
    def load_image(file, label):
        img = np.load(file.numpy().decode('utf-8'))  # load image
        img = standardize(img)                       # standardize image
        img = tf.convert_to_tensor(img, np.float32)  # convert to tensor 
        img = tf.expand_dims(img, axis=2)            # adicionando canal 1 = (182,218,1)
        img = tf.image.grayscale_to_rgb(img)         # convertendo para 3 canais = (182,218,3)
        return img, label

    # Wraps a python function into a TensorFlow op
    def generator_preprocessing(file, labels):
        return tf.py_function(load_image, [file, labels], [tf.float32, tf.float32])


# -------------------------------------------------------------------------------------------------------- #
#                                                DS LOADERS
# -------------------------------------------------------------------------------------------------------- #

    train_loader = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    validation_loader = tf.data.Dataset.from_tensor_slices((validation_files, validation_labels))

# -------------------------------------------------------------------------------------------------------- #
#                                            TRAIN AND VAL DATASETS
# -------------------------------------------------------------------------------------------------------- #

    batch_size = batch_size
    train_dataset = (
        train_loader.shuffle(len(train_files)) # ler sobre isso!
        .map(generator_preprocessing)
        .batch(batch_size)
        .prefetch(batch_size))

    val_dataset = (
        validation_loader.shuffle(len(validation_files))
        .map(generator_preprocessing)
        .batch(batch_size)
        .prefetch(batch_size))
    return train_dataset, val_dataset