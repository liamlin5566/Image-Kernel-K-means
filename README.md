# Image-Kernel-K-means

### Basic Information
This project is the implementation of Kernel K-means on image with c++ and python. I will also try to accelerate the computation on GPU by using CUDA if time is enough.

### Problem to Solve
I will do clustering on image data by using kernel kmeans, which "similar" pixels would be grouped together. 

Kernel k-means is well-known unsupervised clustering algorithm. It will find k cluster centers to represent each group, and assign each data point to its nearest center like traditional k-means. Different from traditional k-means using euclidean distance, kernel kmeans utilize kernel function to calculate distance:

$$\begin{aligned}
\min_{k} (||\phi(x_j) - \mu_{k}||) = \min_{k} (||\phi(x_j) - \sum_{n} {\alpha_{k}^{n} \phi(x_n)} ||)
\end{aligned}
$$

where $x$ represent data point which consists of pixel value (rgb) and posistion, $\phi$ means kernel function, $\mu$ means cluster center, and $alpha$ is assignment with the value 0 or 1. kernel kmeans would search the appropriate assignment that cause minimum distance.


### Requirements
-   opencv4
-   openmp
-   pybind11
-   numpy
-   pytest
-   cuda (optional)

### Compile
```
make all
```
**With cuda (optional)**
```
make wcuda=true
```

### Test
```
make test # pytest
python example.py
```

input image:
<p align="center">
  <img src="./proj/image1.png" />
</p>

result image:
<p align="center">
  <img src="./proj/result/image1_result.png" />
</p>


#### Time
Test 5 times on image1.png

| mode         |  Avg. Time      | 
| -------------| --------------- | 
| cpu (1)      |  21.85 (sec)    |
| cpu (2)      | 12.81  (sec)    |
| cpu (4)      | 6.98  (sec)     |
| gpu          |17.72 (sec)      |
