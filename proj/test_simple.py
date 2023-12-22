import _kmeans
import numpy as np
import pytest
import cv2

def trans2image(labels, img, k):

    h, w = img.shape[0], img.shape[1]
    output = np.zeros((h, w, 3))
    for cls in range(k):
        idxs = np.where(labels == cls)[0]

        mean_color = 0.0
        for idx in idxs:
            y = idx // w
            x = idx % w
            #print(idx)
            mean_color += (img[y][x] / len(idxs))

        #print(mean_color.shape)
        for idx in idxs:
            y = idx // w
            x = idx % w
            output[y, x, :] = mean_color
    
    return output.astype(np.uint8)
    



def test_unit():
    k = 2
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

    # print(c.k_cluster)
    # print(c.gamma_c)
    # print(c.gamma_s)
    # print(c.max_iter)
    # print(c.nthreads)
    # print(c.thresh)


    #print("---------------------------------------")

    img = cv2.imread("./test.png")
    c.predict(img)
    #c.savefig("image1_result.png")

    result = c.get_results()
    assert result.shape[0] == img.shape[0] * img.shape[1]

    #print(result)
    output = trans2image(result, img, k)
    #print((output - img).sum())
    assert (np.abs(output - img)).sum() == 0.0


if __name__ == "__main__":
    test_unit()