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


#ifdef __NVCC__
#pragma message( "Use Cuda")      
#endif

namespace pybind11
{

template<typename DType>
class pyKmeans{
    public:

    pyKmeans(int k, double gamma_c=0.6, double  gamma_s=0.4, int max_iter=10, double  thresh=0.0001, int nthreads=4):
    m_cluster{k, DType(gamma_c), DType(gamma_s), max_iter, DType(thresh), nthreads},
    m_imgdata{}
    {

    }

    void predict(std::string filename)
    {
        set_data(filename);
        #ifndef __NVCC__
            m_cluster.fit(m_imgdata);
        #else
            //std::cout << "use cuda" << std::endl;
            m_cluster.fit_cuda(m_imgdata);
        #endif
    }

    void predict(pybind11::array_t<unsigned char>& np_img)
    {
        if (np_img.ndim() != 3)
            throw std::runtime_error("The channel of input image must be 3\n");

        pybind11::buffer_info buf = np_img.request();

        cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*) buf.ptr);
        m_imgdata.set(img);
        #ifndef __NVCC__
            //std::cout << "use" << std::endl;
            m_cluster.fit(m_imgdata);
        #else
            //std::cout << "use cuda" << std::endl;
            m_cluster.fit_cuda(m_imgdata);
        #endif
    }

    void savefig(std::string filename)
    {
        if (m_cluster.alpha.size() == 0)
            throw std::runtime_error("Have not do clustering, the size of result is zero\n");
        m_cluster.save_fig(m_imgdata, filename);
    }

    pybind11::array_t<int> get_results()
    {
        if (m_cluster.alpha.size() == 0)
            throw std::runtime_error("Have not do clustering, the size of result is zero\n");

        auto np_alpha = pybind11::array_t<int>(m_cluster.alpha.size());
        pybind11::buffer_info buf = np_alpha.request();
        int *ptr = static_cast<int *>(buf.ptr);
        for(std::size_t i = 0; i < m_cluster.alpha.size(); i++){
            ptr[i] = m_cluster.alpha[i];
        }
        return np_alpha;
    }
    
    void set_verbose(bool verbose) {m_cluster.set_verbose(verbose); }

    int k_cluster() {return m_cluster.k_cluster();}
    double gamma_c() {return double(m_cluster.gamma_c());}
    double gamma_s() {return double(m_cluster.gamma_s());}
    int max_iter() {return m_cluster.max_iter();}
    int nthreads() {return m_cluster.nthreads();}
    double thresh() {return double(m_cluster.thresh());}

    private:
    kmeans<DType> m_cluster;
    imagedata<DType> m_imgdata;
    

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
    pybind11::class_<pyKmeans<double>>(m, "kmeans")
        .def_property_readonly("k_cluster", &pyKmeans<double>::k_cluster)
        .def_property_readonly("gamma_c", &pyKmeans<double>::gamma_c)
        .def_property_readonly("gamma_s", &pyKmeans<double>::gamma_s)
        .def_property_readonly("max_iter", &pyKmeans<double>::max_iter)
        .def_property_readonly("nthreads", &pyKmeans<double>::nthreads)
        .def_property_readonly("thresh", &pyKmeans<double>::thresh)
        .def(pybind11::init<std::size_t>()) 
        .def(pybind11::init<std::size_t, double, double>()) 
        .def(pybind11::init<std::size_t, double, double, std::size_t>()) 
        .def(pybind11::init<std::size_t, double, double, std::size_t, std::size_t>()) 
        .def(pybind11::init<std::size_t, double, double, std::size_t, double, std::size_t>()) 
        .def("predict", static_cast<void (pyKmeans<double>::*)(std::string filename)>(&pyKmeans<double>::predict), "input filepath")
        .def("predict", static_cast<void (pyKmeans<double>::*)(pybind11::array_t<unsigned char>&)>(&pyKmeans<double>::predict), "input cv Mat image ")
        .def("savefig", &pyKmeans<double>::savefig)
        .def("get_results", &pyKmeans<double>::get_results)
        .def("set_verbose", &pyKmeans<double>::set_verbose);

    
    pybind11::class_<pyKmeans<float>>(m, "kmeans32f")
        .def_property_readonly("k_cluster", &pyKmeans<float>::k_cluster)
        .def_property_readonly("gamma_c", &pyKmeans<float>::gamma_c)
        .def_property_readonly("gamma_s", &pyKmeans<float>::gamma_s)
        .def_property_readonly("max_iter", &pyKmeans<float>::max_iter)
        .def_property_readonly("nthreads", &pyKmeans<float>::nthreads)
        .def_property_readonly("thresh", &pyKmeans<float>::thresh)
        .def(pybind11::init<std::size_t>()) 
        .def(pybind11::init<std::size_t, double, double>()) 
        .def(pybind11::init<std::size_t, double, double, std::size_t>()) 
        .def(pybind11::init<std::size_t, double, double, std::size_t, std::size_t>()) 
        .def(pybind11::init<std::size_t, double, double, std::size_t, double, std::size_t>()) 
        .def("predict", static_cast<void (pyKmeans<float>::*)(std::string filename)>(&pyKmeans<float>::predict), "input filepath")
        .def("predict", static_cast<void (pyKmeans<float>::*)(pybind11::array_t<unsigned char>&)>(&pyKmeans<float>::predict), "input cv Mat image ")
        .def("savefig", &pyKmeans<float>::savefig)
        .def("get_results", &pyKmeans<float>::get_results)
        .def("set_verbose", &pyKmeans<float>::set_verbose);

}