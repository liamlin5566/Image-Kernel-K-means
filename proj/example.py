import _kmeans
c = _kmeans.kmeans(3)

print(c.k_cluster)
print(c.gamma_c)
print(c.gamma_s)
print(c.max_iter)
print(c.nthreads)

c.predict_and_savefig("./image1.png")