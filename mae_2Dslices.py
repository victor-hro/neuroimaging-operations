# ======================================================================================================== #
# ============================================   Show Results   ========================================== #
# ======================================================================================================== #

#--------------------------------------------------------------------------------------------------------- #
#                                      função que retorna o erro absoluto  
#--------------------------------------------------------------------------------------------------------- #

#  entrada = imagem 3D
#  saída   = idade cerebral


def model_evaluate(model, img, label):
    pred_list = [] # predictions
    #ae_list   = [] # absolute error
    
    # load image
    img = nib.load(img)
    img = img.get_fdata()
    img = standardize(img)
    img = tf.convert_to_tensor(img, np.float32)
    
    for slice in range(60):
        # 2D image
        clipped = img[:,:,slice+10]
        clipped = tf.expand_dims(clipped, axis=2)            # adicionando canal 1 = (:,:,1)
        clipped = tf.image.grayscale_to_rgb(clipped)         # convertendo para 3 canais = (:,:,3)
        clipped = tf.expand_dims(clipped, axis=0)            # adicionando canal 1 = (1,:,:,3)
        
        # os valores armazenados em pred_list servem para cálculo da mediana
        # cada slice é previsto. No final a mediana é calculada,
        # retornando com a idade prevista pelo modelo.
        pred = model.predict(clipped)
        pred_list.append(pred)
        
        # calcula o erro absoluto por imagem
        #ae = np.abs(pred - label)
        #ae_list.append(ae)
    
    median_ = np.median(pred_list)
    ae = np.abs(median_ - label)
    
    #print(f'Idade prevista: {median_} || Idade real: {label}\n')
    return ae, median_, label

#--------------------------------------------------------------------------------------------------------- #
#                                            função que retorna o MAE  
#--------------------------------------------------------------------------------------------------------- #

#  ----------------------------------------------
#  entrada: model, quantidade de pacientes
#  saída:   MAE
#  ----------------------------------------------
#  ae_list:     absolute error array
#  pred_list:   array contendo todas as previsões
#  labels_list: array contendo os rótulos
#  ----------------------------------------------


def show_results(model, n_files):
    
    ae_list     = []
    pred_list   = np.arange(n_files)
    labels_list = np.arange(n_files)
    
    for c in range(n_files):
        
        # ae recebe o absolute error = np.abs(median_ - label)
        ae, median_, label = model_evaluate(model, files_3d[c], labels_3d[c])
        
        ae_list.append(ae)
        #pred_list.append(median_)
        #labels_list.append(label)
        pred_list[c]   = median_
        labels_list[c] = label
       
    # cálculo do mae
    mae = np.sum(ae_list) / n_files
    print(f'MAE: {mae}')
     
    # plotando os valores
    idx = labels_list.argsort()
    plt.plot(labels_list[idx], labels_list[idx], 'b-')
    plt.plot(labels_list[idx], pred_list[idx], 'r.')
    plt.title('Valor previsto x Valor real')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.grid()

    plt.tight_layout()
    return mae