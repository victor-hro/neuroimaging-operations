def gen_dataset(files: str, labels: float, batch_size, augmentation=False):
    """
    Args:
        files: file paths
        labels: ages
        augmentation = dict of augmentation options
    """
    def prepare_image_(filename, label):
        # load image from disk
        img = np.load(filename.numpy().decode("utf-8"))

        # data augmentation
        if augmentation:
            transform = A.Compose(augmentation.values())
            img = transform(image=img)["image"]

        # convert to tensor
        img = tf.convert_to_tensor(img, np.float32)
        # add last channel = (w, h, 1)
        img = tf.expand_dims(img, axis=-1)
        # convert to 3 channels = (w, h, 3)
        img = tf.image.grayscale_to_rgb(img)
        return img, label

    # Wraps a python function into a TensorFlow op
    def prepare_image(file, label):
        return tf.py_function(prepare_image_, [file, label], [tf.float32, tf.float64])

    dataset = (
        tf.data.Dataset.from_tensor_slices((files, labels))
        .shuffle(len(files))
        .map(prepare_image)
        .batch(batch_size)
        .prefetch(batch_size)
    )
    return dataset