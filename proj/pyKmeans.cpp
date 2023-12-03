#include "include/imagedata.hpp"
#include "include/kmeans.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py=pybind11;

class pyKmeans{
    public:

    pyKmeans(int k,  double gamma_c=0.6, double gamma_s=0.4):
    m_cluster{k, gamma_c, gamma_s, 1, 6},
    m_imgdata{}
    {

    }

    void predict_and_savefig(std::string filename)
    {
        set_data(filename);
        m_cluster.fit(m_imgdata);
        m_cluster.save_fig(m_imgdata, "A.png");
    }

    int k_cluster() {return m_cluster.k_cluster();}
    double gamma_c() {return m_cluster.gamma_c();}
    double gamma_s() {return m_cluster.gamma_s();}
    int max_iter() {return m_cluster.max_iter();}
    int nthreads() {return m_cluster.nthreads();}

    private:
    imagedata m_imgdata;
    kmeans m_cluster;

    void set_data(std::string filename)
    {
        cv::Mat img = cv::imread(filename);
        m_imgdata.set(img);
    }

};


PYBIND11_MODULE(_kmeans, m) 
{
    m.doc() = "K-means on image";

    py::class_<pyKmeans>(m, "kmeans")
        .def_property_readonly("k_cluster", &pyKmeans::k_cluster)
        .def_property_readonly("gamma_c", &pyKmeans::gamma_c)
        .def_property_readonly("gamma_s", &pyKmeans::gamma_s)
        .def_property_readonly("max_iter", &pyKmeans::max_iter)
        .def_property_readonly("nthreads", &pyKmeans::nthreads)
        .def(py::init<std::size_t>()) 
        .def(py::init<std::size_t, double, double>()) 
        .def("predict_and_savefig", &pyKmeans::predict_and_savefig);

}