def predictions_dataframe(X: str, y_true: float, y_pred: float):
    """
    Args:
        X: filename
        y_true: true age
        y_pred: predicted age
    Returns:
        Dataframe of predictions and PAD(Predict Age Diff)
    """
    # round using .:2f
    format_ = lambda x: "%.2f" % x

    # create dataframe
    df = pd.DataFrame(
        {
            "filename": X,
            "age": y_true,
            "prediction": y_pred,
            "PAD": mean_absolute_error(y_true, y_pred),
        }
    )

    # aplying roundying
    df["prediction"] = df["prediction"].map(format_)
    df["PAD"] = df["PAD"].map(format_)
    return df