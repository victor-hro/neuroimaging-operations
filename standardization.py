class Standardize():
   
   def __init__(self, img):
      
      self.img = img

   def standardize(self):
      img[img < 0] = 0
      mask = img > 0 # bool
      mean = np.mean(img[mask]) # mean
      std = np.std(img[mask])   # std

      img[mask] = (img[mask] - mean)/ std
      return img