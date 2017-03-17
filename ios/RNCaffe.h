#import <UIKit/UIKit.h>
#import <React/RCTBridgeModule.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
using namespace caffe;
using namespace std;

/* Pair (label, confidence) representing a prediction. */
typedef pair<string, float> Prediction;

@interface RNCaffe : NSObject <RCTBridgeModule>
+ (bool) saveImage: (UIImage *)image path:(NSString *)path;
+ (UIImage *)resize:(UIImage*)image newSize:(CGSize)newSize;
+ (bool) copyAssetImage:(NSString*)uri path:(NSString*)path;
@end

@interface Classifier : NSObject {
@private
  shared_ptr<Net<float> > net_;
  int num_channels_;
  vector<string> labels_;
  cv::Size input_geometry_;
  cv::Mat mean_;
}
//-(id)init; //不带参数的构造函数
-(id)initWithModel:(const string&)model
           weight:(const string&)weight
              mean:(const string&)mean
             label:(const string&)label;   //带参数的构造函数
-(id)initWithModel:(const string&)model
            weight:(const string&)weight
              mean:(const string&)mean;   //带参数的构造函数
-(id)initWithModel:(const string&)model
            weight:(const string&)weight;   //带参数的构造函数
- (vector<Prediction>) Classify:(const cv::Mat&) img;
- (vector<Prediction>) Classify:(const cv::Mat&) img  N:(int) N;
- (vector<float>) Predict:(const cv::Mat&) img;
- (void) SetMean:(const string&) mean_file;
- (void) WrapInputLayer:(vector<cv::Mat>*) input_channels;
- (void) Preprocess:(const cv::Mat&) img
     input_channels:(vector<cv::Mat>*) input_channels;

@end
