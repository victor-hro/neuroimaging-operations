class LinearDecay:
    def __init__(
        self,
        max_epochs=10,
        min_lr=1e-3,
        max_lr=0
    ):
        
        # lr inicial
        self.min_lr = min_lr
        # lr final
        self.max_lr = max_lr
        self.max_epochs = max_epochs

    def __call__(self, epoch):
        decay = epoch / self.max_epochs

        # epoch 0 = min_lr
        lr = self.min_lr + (self.max_lr - self.min_lr) * decay
        return lr