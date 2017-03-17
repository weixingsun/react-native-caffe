#ifndef MCVCAFFE_H
#define MCVCAFFE_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//#include <glibmm.h>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class  McvWin;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);
  ~Classifier();
 
  std::vector<Prediction> Classify(const cv::Mat& img, int_tp N = 5);

    void  load_image(cv::Mat &img);
    void  signal_connect(McvWin  *p_win);
    void  get_str_class(std::string  &str_class);
    void  fun();

private:
    std::unique_ptr<std::thread>         mptr_thread;
    Glib::Dispatcher                     signal_thread;
    cv::Mat                              m_img;
    std::vector<Prediction>              m_predictions;
    std::string                          m_str_class;
    std::mutex                           m_mutex;
    //std::thread
    
 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int_tp num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif // MCVCAFFE_H
