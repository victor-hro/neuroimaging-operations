# =========================================================================================================== #
# ========================================== LEARNING RATE MONITOR ========================================== #
# =========================================================================================================== #


class LearningRateMonitor(Callback):
    # end of each training epoch
    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        # get and store the learning rate
        lrate = float(backend.get_value(self.model.optimizer.lr))
        logs["learning_rate"] = lrate

        return