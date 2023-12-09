CXX = g++ -O3 -Wall -std=c++11 -fopenmp -fPIC 
OPENCV = $$(pkg-config --cflags --libs opencv4)
LIB_TARGET = _kmeans.so

# all: main.cpp build/kmeans.o build/imagedata.o
# 	$(CXX) $^ -o main.out $(OPENCV) 
all: $(LIB_TARGET)

$(LIB_TARGET): pyKmeans.cpp build/kmeans.o build/imagedata.o
	$(CXX) $^  -shared  `python3 -m pybind11 --includes` -o $@ $(OPENCV) 

build/kmeans.o: src/kmeans.cpp include/kmeans.hpp build/imagedata.o
	mkdir -p build
	$(CXX) -c src/kmeans.cpp  -I./include -o build/kmeans.o

build/imagedata.o: src/imagedata.cpp include/imagedata.hpp
	mkdir -p build
	$(CXX) -c src/imagedata.cpp  -I./include -o build/imagedata.o

test:
	python3 -m pytest test_simple.py -v

clean:
	rm -r -f build *.so