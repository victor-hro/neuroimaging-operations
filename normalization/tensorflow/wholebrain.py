def standardize(self):
   img[img < 0] = 0
   mask = img > 0 # bool
   mean = np.mean(img[mask]) # mean
   std = np.std(img[mask])   # std

   img = (img - mean)/ std
   return img
