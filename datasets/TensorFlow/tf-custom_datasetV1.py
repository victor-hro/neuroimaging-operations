class MyDataset(keras.utils.Sequence):
    def __init__(self, data_encoder,batch_size=8, dim = (182, 218, 182), n_channels = 1, shuffle = True):
        
        self.batch_size = batch_size
        self.dim        = dim
        self.n_channels = n_channels
        self.encoder    = data_encoder
        self.labels     = data_encoder[:,1]          # labels
        self.list_IDs   = data_encoder[:,0]          # caminho das imagens
        self.shuffle    = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size)) # retorna o tamanho do batch
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.encoder[k] for k in indexes] # recebe o encoder aleatório, conforme o index
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))  # shape = (n_samples, 182, 218, 182,1)
        y = np.empty((self.batch_size,), dtype=float)                 # shape = (n_samples,)

        # Generate data
        for i, ID in enumerate(list_IDs_temp): # ID recebe cada diretório
            # Store sample
            xx = np.load(ID[0]).astype(np.float32) 
            xx = standardize(xx)
            xx = tf.expand_dims(xx, axis=3)
            X[i,] = xx                             # linha i, coluna dos caminhos das imagens

            # Store class
            y[i] = ID[1]                           # linha i, coluna das labels

        return X, y