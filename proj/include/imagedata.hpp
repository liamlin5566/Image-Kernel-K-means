#ifndef IMAGEDATA_HPP
#define IMAGEDATA_HPP
#include <opencv2/opencv.hpp>
#include <vector>

template<typename T>
class imagedata{
public:
    imagedata()
    {
       
    }
    imagedata(cv::Mat &img)
    {
        m_img = img;
        m_height = img.rows;
        m_width = img.cols;
        img2inputdata();
    }

    void set(cv::Mat &img)
    {
        m_img = img;
        m_height = img.rows;
        m_width = img.cols;
        img2inputdata();
    }
    
    std::size_t height() {return m_height; }
    std::size_t width() {return m_width; }
    std::size_t size() {return m_data.size(); }

    std::vector<T>  operator[] (std::size_t index) const
    {
        return m_data[index];
    }
    std::vector<T> & operator[] (std::size_t index)
    {
        return m_data[index];
    }

private:
    cv::Mat m_img;
    std::size_t m_height;
	std::size_t m_width;
    std::vector<std::vector<T>> m_data;

    void img2inputdata();
}; 

#endif