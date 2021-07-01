# ============================================================================================================ #
# =========================================   MEAN ABSOLUTE ERROR    ========================================= #
# ============================================================================================================ #


def show_results(
    model, X, y,
    plot=False, plot_mse=False,
    list_errors=False, subset=""):
    
    """
    Args:
        X: path of images
        y: labels
        list_errors: list of all absolute errors
        plot = plot the regression graph (label, predict) and returns X, y, predictions list and absolut error list
        plot_mse = plot two regression graph ([mse, rmse], slice) and returns mse, rmse list.
    
    """

    ae_list = np.arange(len(X), dtype=float)
    pred_list = np.arange(len(X), dtype=float)
    labels_list = np.arange(len(y), dtype=float)

    # loop para ler todas as imagens e os labels
    for c in range(len(X)):
        # ae recebe o absolute error = np.abs(median_ - label)
        ae, median_, label = model_predict(model, X[c], y[c])

        ae_list[c] = ae
        pred_list[c] = median_
        labels_list[c] = label

    # cálculo do mae
    mae = np.mean(ae_list)
    print(f" - {subset} mae: {round(mae,2)}")

    if plot == True:
        # plotando os valores
        idx = labels_list.argsort()
        plt.plot(labels_list[idx], labels_list[idx], "b-")
        plt.plot(labels_list[idx], pred_list[idx], "r.")
        plt.title("Valor previsto x Valor real")
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.grid()

        plt.tight_layout()

    if list_errors == True:
        return X, y, pred_list, ae_list
    return mae


def model_predict(model, img, label):
    N_slices = 55
    initial_slice = 15
    # matriz que irá armazenar todos os slices e o label
    img_array = np.empty((N_slices, 91, 109, 3), dtype=float)
    labels_array = np.empty((N_slices,), dtype=float)

    # operações com a imagem
    img = nib.load(img)
    img = img.get_fdata()

    # divisão em 80 slices 2D
    for slice in range(N_slices):
        clipped = img[:, :, slice + initial_slice]
        clipped = norm_percentile(clipped)
        clipped = standardize_eval(clipped)  # standardize image
        clipped = tf.convert_to_tensor(clipped, np.float32)  # convert to tensor
        clipped = tf.expand_dims(clipped, axis=2)  # adicionando canal 1 = (182,218,1)
        clipped = tf.image.grayscale_to_rgb(
            clipped
        )  # convertendo para 3 canais = (182,218,3)

        # armazena slice por slice
        img_array[slice,] = clipped
        labels_array[slice] = label

    evaluation_loader = tf.data.Dataset.from_tensor_slices((img_array, labels_array))
    evaluation_dataset = evaluation_loader.batch(N_slices).prefetch(N_slices)

    # faz a predição do batch
    pred = model.predict(evaluation_dataset)
    median_ = np.median(pred)
    ae = np.abs(median_ - label)

    return ae, median_, label


# =========================================================================================================== #
# ============================================= VAL MAE CALLBACK ============================================ #
# =========================================================================================================== #


class CustomCallback(keras.callbacks.Callback):
    """
    Args:
        X = list of filenames
        y = labels
        subset = training, validation or test

    Returns:
        Subset MAE
    """

    def __init__(self, X, y, subset=""):
        self.X = X
        self.y = y
        self.subset = subset

    def on_train_begin(self, logs=None):
        # best prediction
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        t1 = time()

        # call the show results function and returns the MAE
        pred = show_results(self.model, self.X, self.y, subset=self.subset)

        # include MAE in logs
        logs["val_mae_vol"] = pred

        t2 = time()
        print(f" - Time: {round((t2-t1)/60,2)} minutos")

        # Record the best weights if current results is better (less).
        if np.less(pred, self.best):
            self.best = pred
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    # storage best weights
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        print(f"Best epoch: {self.best_epoch + 1}, val_mae_vol: {self.best}")