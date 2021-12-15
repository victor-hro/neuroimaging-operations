def model_predict(img):
    """ Predict the image
    Args: 3D MRI image: np.array
    Returns: predicted age: float32
    """
    # transforms to numpy if the input is a tensor
    if not type(img) == np.ndarray is False:
        img = img[0].numpy()

    # normalization:
    # if the image is not normalized, we normalize it
    # use the same normalization that we used to train the model
    # we normalize the image by substracting the mean and dividing by the standard deviation
    # we also need to convert the image to float32
    # we also need to convert the image to a tensor
    # we also need to expand the dimension of the image

    img = norm_percentile(img)
    img = standardize(img)

    # shape = (w, h, all_slices)
    img = img[
        :, :, initial_slice : initial_slice + n_slices
    ]  # shape = (w, h, n_slices)
    img = np.moveaxis(img, -1, 0)  # shape = (n_slices, w, h)
    img = tf.expand_dims(img, axis=-1)  # shape = (n_slices, w, h, 1)
    img = tf.image.grayscale_to_rgb(img)  # shape = (n_slices, w, h, 3)
    img = tf.convert_to_tensor(img, np.float32)

    # print("tensor ops")
    ages = model.predict(img)
    # print("predicted")
    age = np.median(ages)
    # print("age")

    return age