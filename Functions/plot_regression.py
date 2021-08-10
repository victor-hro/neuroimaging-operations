def plot_regression(y_true: float, y_pred: float):
    # mean_absolute_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    y_true = np.array(y_true)
    idx = np.argsort(y_true)

    plt.plot(y_true[idx], y_true[idx], "b-")
    plt.plot(y_true[idx], y_pred[idx], "r.")
    plt.title(f"Valor previsto x Valor real\nMean absolute error: {(mae):.2f}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.legend(["real", "predict"])
    plt.grid()
    plt.tight_layout()