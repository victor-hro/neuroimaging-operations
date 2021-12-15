def predict_ages(model, files: str):
    """
    Args:
        model: The model to use for prediction
        files: The path to the image files
    Returns:
        A list containing the predicted age for each image
    """

    # creates a dataset from the files
    dataset = (
        tf.data.Dataset.from_tensor_slices(files).map(load_image)
        # .map(model_predict)
        # .prefetch(tf.data.experimental.AUTOTUNE)
        .prefetch(1)
    )

    # creates a iterator for the dataset
    ages = np.zeros(len(files), dtype=float)

    # iterates over the dataset
    for i, filename in enumerate(dataset):
        # call the model_predict function
        ages[i] = model_predict(filename)
    return ages