# =========================================================================================================== #
# ============================================= INTRA EPOCH MAE CALLBACK ==================================== #
# =========================================================================================================== #


class LossIntraEpoch(keras.callbacks.Callback):
    """
    Returns:
        Loss per batch
    """
    def on_epoch_begin(self,epoch, logs=None):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs={}):
        self.per_batch_losses.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs=None):
        logs["per_batch_losses"] = self.per_batch_losses
