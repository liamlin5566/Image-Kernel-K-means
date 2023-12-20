#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include <opencv2/opencv.hpp>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "include/imagedata.hpp"
#include "include/kmeans.hpp"

namespace pybind11
{

namespace py=pybind11;
class pyKmeans{
    public:

    pyKmeans(int k,  double gamma_c=0.6, double gamma_s=0.4, int max_iter=10, double thresh=0.0001, int nthreads=4):
    m_cluster{k, gamma_c, gamma_s, max_iter, thresh, nthreads},
    m_imgdata{}
    {

    }

    void predict(std::string filename)
    {
        set_data(filename);
        #ifndef WCUDA
            //std::cout << "use" << std::endl;
            m_cluster.fit(m_imgdata);
        #else
            //std::cout << "use cuda" << std::endl;
            m_cluster.fit_cuda(m_imgdata);
        #endif
    }

    void predict(py::array_t<unsigned char>& np_img)
    {
        if (np_img.ndim() != 3)
            throw std::runtime_error("The channel of input image must be 3\n");

        py::buffer_info buf = np_img.request();

        cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*) buf.ptr);
        m_imgdata.set(img);
        #ifndef WCUDA
            //std::cout << "use" << std::endl;
            m_cluster.fit(m_imgdata);
        #else
            //std::cout << "use cuda" << std::endl;
            m_cluster.fit_cuda(m_imgdata);
        #endif
    }

    void savefig(std::string filename)
    {
        if (m_cluster._alpha.size() == 0)
            throw std::runtime_error("Have not do clustering, the size of result is zero\n");
        m_cluster.save_fig(m_imgdata, filename);
    }

    py::array_t<int> get_results()
    {
        if (m_cluster._alpha.size() == 0)
            throw std::runtime_error("Have not do clustering, the size of result is zero\n");

        auto np_alpha = py::array_t<int>(m_cluster._alpha.size());
        py::buffer_info buf = np_alpha.request();
        int *ptr = static_cast<int *>(buf.ptr);
        for(std::size_t i = 0; i < m_cluster._alpha.size(); i++){
            ptr[i] = m_cluster._alpha[i];
        }
        return np_alpha;
    }
    
    void set_verbose(bool verbose) {m_cluster.set_verbose(verbose); }

    int k_cluster() {return m_cluster.k_cluster();}
    double gamma_c() {return m_cluster.gamma_c();}
    double gamma_s() {return m_cluster.gamma_s();}
    int max_iter() {return m_cluster.max_iter();}
    int nthreads() {return m_cluster.nthreads();}
    double thresh() {return m_cluster.thresh();}

    private:
    kmeans m_cluster;
    imagedata m_imgdata;
    

    void set_data(std::string filename)
    {
        cv::Mat img = cv::imread(filename);
        m_imgdata.set(img);
    }

};

}/*end namespace pybind11*/


PYBIND11_MODULE(_kmeans, m) 
{
    m.doc() = "K-means on image";

    using namespace pybind11;
    py::class_<pyKmeans>(m, "kmeans")
        .def_property_readonly("k_cluster", &pyKmeans::k_cluster)
        .def_property_readonly("gamma_c", &pyKmeans::gamma_c)
        .def_property_readonly("gamma_s", &pyKmeans::gamma_s)
        .def_property_readonly("max_iter", &pyKmeans::max_iter)
        .def_property_readonly("nthreads", &pyKmeans::nthreads)
        .def_property_readonly("thresh", &pyKmeans::thresh)
        .def(py::init<std::size_t>()) 
        .def(py::init<std::size_t, double, double>()) 
        .def(py::init<std::size_t, double, double, std::size_t>()) 
        .def(py::init<std::size_t, double, double, std::size_t, std::size_t>()) 
        .def(py::init<std::size_t, double, double, std::size_t, double, std::size_t>()) 
        .def("predict", static_cast<void (pyKmeans::*)(std::string filename)>(&pyKmeans::predict), "input filepath")
        .def("predict", static_cast<void (pyKmeans::*)(py::array_t<unsigned char>&)>(&pyKmeans::predict), "input cv Mat image ")
        .def("savefig", &pyKmeans::savefig)
        .def("get_results", &pyKmeans::get_results)
        .def("set_verbose", &pyKmeans::set_verbose);

}