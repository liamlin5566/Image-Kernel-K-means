import _kmeans
import numpy as np
#import cv2
gamma_c=0.6
gamma_s=0.4
max_iter=20
thresh=0.001
nthreads=4

c = _kmeans.kmeans(3, gamma_c, gamma_s, max_iter, thresh, nthreads)
c.set_verbose(True)
print(c.k_cluster)
print(c.gamma_c)
print(c.gamma_s)
print(c.max_iter)
print(c.nthreads)
print(c.thresh)

# c.predict("./test.png")
# c.savefig("A.png")

print("---------------------------------------")

#img = cv2.imread("./image1.png")
c.predict("./image1.png")
c.savefig("image1_result.png")

reult = c.get_results()
print(reult.shape)