import numpy as np
import PIL  
from PIL import Image  


img = np.zeros((20,20), dtype=np.uint8)
img[:, :10] = 255

pli_img = Image.fromarray(img)
pli_img.save( "test.png" )

