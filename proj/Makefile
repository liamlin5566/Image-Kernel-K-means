CXX = g++
OPENCV = $$(pkg-config --cflags --libs opencv4)


all: main.cpp build/kmeans.o build/imagedata.o
	$(CXX) $^ -o main.out $(OPENCV) 

build/kmeans.o: src/kmeans.cpp include/kmeans.hpp build/imagedata.o
	mkdir -p build

	$(CXX) -c src/kmeans.cpp  -I./include -o build/kmeans.o


build/imagedata.o: src/imagedata.cpp include/imagedata.hpp
	mkdir -p build

	$(CXX) -c src/imagedata.cpp  -I./include -o build/imagedata.o


clean:
	rm -r -f build