	 
export CPATH=$CPATH:/usr/local/cuda/include

CXX = g++ -O3 -Wall -std=c++11 -fopenmp -fPIC 
NVCC = nvcc -O3 -std=c++11 -Xcompiler -fopenmp -Xcompiler -fPIC #N -x cu 

OPENCV = $$(pkg-config --cflags --libs opencv4)
LIB_TARGET = _kmeans.so



all: $(LIB_TARGET)

ifneq ($(wcuda), true)
$(LIB_TARGET): pyKmeans.cpp build/kmeans.o build/imagedata.o 
	$(CXX) $^  -shared  `python3 -m pybind11 --includes` -o $@  $(OPENCV) 

build/kmeans.o: src/kmeans.cpp include/kmeans.hpp build/imagedata.o
	mkdir -p build
	$(CXX) -c src/kmeans.cpp  -I./include -o build/kmeans.o

build/imagedata.o: src/imagedata.cpp include/imagedata.hpp
	mkdir -p build
	$(CXX) -c src/imagedata.cpp  -I./include -o build/imagedata.o

else
	@echo "$(wcuda)"
$(LIB_TARGET): pyKmeans.cpp build/kmeans.o build/imagedata.o build/kernelfunc.o
	$(NVCC) $^  -shared  `python3 -m pybind11 --includes` -o $@ -L /usr/local/cuda/lib64 -lcudart -lcuda  $(OPENCV) 

build/kmeans.o: src/kmeans.cpp include/kmeans.hpp build/imagedata.o build/kernelfunc.o
	mkdir -p build
	$(NVCC) -x cu -c src/kmeans.cpp  -I./include -I./include -o build/kmeans.o 

build/imagedata.o: src/imagedata.cpp include/imagedata.hpp
	mkdir -p build
	$(CXX) -c src/imagedata.cpp  -I./include -o build/imagedata.o

build/kernelfunc.o: src/kernelfunc.cu include/kernelfunc.h
	mkdir -p build
	$(NVCC) -c src/kernelfunc.cu  -I./include  -o build/kernelfunc.o 
endif


test:
	@echo "$(wcuda)"
	python3 -m pytest test_simple.py 


cuda:

clean:
	rm -r -f build *.so