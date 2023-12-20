import _kmeans
import numpy as np
import pytest
import cv2
import time

def cluster():
    k = 3
    gamma_c=0.6
    gamma_s=0.4
    max_iter=20
    thresh=0.001
    nthreads=4

    c = _kmeans.kmeans(k, gamma_c, gamma_s, max_iter, thresh, nthreads)
    c.set_verbose(True)

    assert c.k_cluster == k
    assert c.gamma_c == gamma_c
    assert c.gamma_s == gamma_s
    assert c.max_iter == max_iter
    assert c.nthreads == nthreads
    assert c.thresh == thresh

    

    print("------------------Do Cluster-----------------")
    c.predict("./image1.png")
    c.savefig("image1_result.png")



def test_time():
    k = 3
    gamma_c=0.6
    gamma_s=0.4
    max_iter=20
    thresh=0.001
    nthreads= 2
    c = _kmeans.kmeans(k, gamma_c, gamma_s, max_iter, thresh, nthreads)

    img = cv2.imread("./image1.png")

    stime = time.time()
    for _ in range(5):
        c.predict(img)
    etime = time.time()
    
    c.savefig("image1_result.png")
    return (etime - stime) / 5



if __name__ == "__main__":
    cluster()

    print(test_time())