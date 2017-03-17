#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <React/RCTBridgeModule.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <numeric>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "RNCaffe.h"
#include "OpenCVUtil.h"

using namespace caffe;
using namespace std;

@implementation RNCaffe

Classifier *classifier;
RCT_EXPORT_MODULE();

RCT_EXPORT_METHOD(setup2:(NSString *)model_file
                  weight_file:(NSString *)weight_file
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
    NSLog(@"use model: %@, \nweight: %@", model_file,weight_file);
    const string& model_str = [model_file UTF8String];
    const string& weight_str= [weight_file UTF8String];
    classifier = [[Classifier alloc] initWithModel:model_str weight:weight_str];
    return resolve(@{@"success": @"yes"});
}
RCT_EXPORT_METHOD(setup4:(NSString *)model_file
                  weight_file:(NSString *)weight_file
                  mean_file:(NSString *)mean_file
                  label_file:(NSString *)label_file
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
    NSLog(@"use model: %@, \nweight: %@, \nmean: %@, \nlabel: %@", model_file,weight_file,mean_file,label_file);
    const string& model_str = [model_file UTF8String];
    const string& weight_str= [weight_file UTF8String];
    const string& mean_str  = [mean_file UTF8String];
    const string& label_str = [label_file UTF8String];
    classifier = [[Classifier alloc] initWithModel:model_str
                                            weight:weight_str
                                              mean:mean_str
                                             label:label_str];
    return resolve(@{@"success": @"yes"});
}
RCT_EXPORT_METHOD(setup3:(NSString *)model_file
                  weight_file:(NSString *)weight_file
                  mean_file:(NSString *)mean_file
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
    NSLog(@"use model: %@, \nweight: %@, \nmean: %@", model_file,weight_file,mean_file);
    const string& model_str = [model_file UTF8String];
    const string& weight_str= [weight_file UTF8String];
    const string& mean_str  = [mean_file UTF8String];
    //const string& label_str = [label_file UTF8String];
    classifier = [[Classifier alloc] initWithModel:model_str
                                            weight:weight_str
                                              mean:mean_str];
    return resolve(@{@"success": @"yes"});
}
RCT_EXPORT_METHOD(seeImage:(NSString *)image_file
             resolver:(RCTPromiseResolveBlock)resolve
             rejecter:(RCTPromiseRejectBlock)reject
                  ){
    //NSLog(@"check image: %@", image_file);
    const string image([image_file UTF8String]);
    cv::Mat img = cv::imread(image, -1);
    vector<Prediction> predictions = [classifier Classify:img];
    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        //cout << fixed << setprecision(4) << p.second << " - \"" << p.first << "\"" << endl;
        NSString *key = [NSString stringWithUTF8String:p.first.c_str()];
        //key = @(p.first);
        NSNumber *value = [NSNumber numberWithFloat:p.second];
        result[key] = value;
    }
    return resolve(result);
}
RCT_EXPORT_METHOD(runModel:(NSString *)model
                  weight:(NSString *)weight
                  image:(NSString *)image
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject
                  ) {
    NSLog(@"use model: %@, \nweight: %@, \nprocess image: %@", model,weight,image);
    //UIImage *image = [UIImage imageWithContentsOfFile:img];
    caffe::Net<float> *_net;
    _net = new caffe::Net<float>([model UTF8String], caffe::TEST);
    _net->CopyTrainedLayersFrom([weight UTF8String]);
    caffe::Blob<float> *input_layer = _net->input_blobs()[0];
    vector<float> mean;
    if(! ReadImageToBlob(image, mean, input_layer)) {
        NSString * errmsg = @"ReadImageToBlob failed";
        return reject(@"error", errmsg, nil);
    }
    _net->Forward();
    caffe::Blob<float> *output_layer = _net->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    vector<float> result(begin, end);
    vector<size_t> result_idx(result.size());
    iota(result_idx.begin(), result_idx.end(), 0);
    sort(result_idx.begin(), result_idx.end(),
              [&result](size_t l, size_t r){return result[l] > result[r];});
    NSMutableArray *arr = [NSMutableArray array];
    for (int i=0; i<result_idx.size(); i++) {
        NSNumber *answer = [NSNumber numberWithLong: result_idx[i]];
        NSNumber *proxibility= [NSNumber numberWithFloat: result[result_idx[i]]];
        //NSString *sa = [answer stringValue];
        [arr addObject:@{
                @"answer":answer,
                @"proxibility":proxibility
        }];
        
    }
    return resolve(arr);
}
RCT_EXPORT_METHOD(faceDetect:(NSString *)path
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    NSData *data = [NSData dataWithContentsOfFile:path];
    UIImage *img = [[UIImage alloc] initWithData:data];
    NSArray *arr = [OpenCVUtil facePointDetectForImage:img];
    NSMutableDictionary *ret = [NSMutableDictionary new];
    BOOL success = YES;
    NSString *errmsg = @"";
    NSMutableArray *marr = [NSMutableArray new];
    for (NSNumber* rectValue in arr) {
        CGRect rect = [rectValue CGRectValue];
        NSDictionary *one = @{
         @"x": @(rect.origin.x),
         @"y": @(rect.origin.y),
         @"w": @(rect.size.width),
         @"h": @(rect.size.height)
        };
        [marr addObject:one];
    }
    [ret setObject:marr forKey:@"rects"];
    [ret setObject:[NSNumber numberWithLong:arr.count] forKey:@"count"];
    if (!success) {
        //errmsg = [NSString stringWithFormat:@"no face detected"];
        return reject(@"error", errmsg, nil);
    }
    return resolve(ret);
}
RCT_EXPORT_METHOD(resizeImage:(NSString *)pathIn
                  output:(NSString *)pathOut
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject){
    NSData *data = [NSData dataWithContentsOfFile:pathIn];
    UIImage *imgIn = [[UIImage alloc] initWithData:data];
    UIImage *imgOut = [RNCaffe resize:imgIn newSize:CGSizeMake(112,112)];
    if([RNCaffe saveImage:imgOut path:pathOut]){
        NSLog(@"OpenCV.resizeImage %@",pathOut);
        return resolve(pathOut);
    }else{
        NSString *errmsg = [NSString stringWithFormat:@"failed to write file"];
        return reject(@"error", errmsg, nil);
    }
}
+ (bool) saveImage: (UIImage *)image path:(NSString *)path {
    NSString *suffix = [[path pathExtension] lowercaseString];
    if([suffix isEqualToString: @"png"]) {
        return [UIImagePNGRepresentation(image) writeToFile:path atomically:YES];
    }else if([suffix isEqualToString: @"jpg"] || [suffix isEqualToString: @"jpeg"]) {
        return [UIImageJPEGRepresentation(image, 1.0) writeToFile:path atomically:YES];
    }
    return false;
}
//asset-library
+ (bool) copyAssetImage:(NSString*)uri path:(NSString*)path{
    ALAssetsLibrary *lib = [[ALAssetsLibrary alloc] init];
    [lib assetForURL:nil resultBlock:nil failureBlock:nil];
    return false;
}
+(UIImage *)resize:(UIImage*)image newSize:(CGSize)newSize {
    CGRect newRect = CGRectIntegral(CGRectMake(0, 0, newSize.width, newSize.height));
    CGImageRef imageRef = image.CGImage;
    
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 0);
    CGContextRef context = UIGraphicsGetCurrentContext();
    
    // Set the quality level to use when rescaling
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGAffineTransform flipVertical = CGAffineTransformMake(1, 0, 0, -1, 0, newSize.height);
    
    CGContextConcatCTM(context, flipVertical);
    // Draw into the context; this scales the image
    CGContextDrawImage(context, newRect, imageRef);
    
    // Get the resized image from the context and a UIImage
    CGImageRef newImageRef = CGBitmapContextCreateImage(context);
    UIImage *newImage = [UIImage imageWithCGImage:newImageRef];
    
    CGImageRelease(newImageRef);
    UIGraphicsEndImageContext();
    
    return newImage;
}
// Read a jpg/png image from file to Caffe input_layer.
// Modified on tensorflow ios example,
// URL: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ios_examples/simple/ios_image_load.mm
bool ReadImageToBlob(NSString *file_name,
                     const vector<float> &mean,
                     caffe::Blob<float>* input_layer) {
    // Get file size
    FILE* file_handle = fopen([file_name UTF8String], "rb");
    fseek(file_handle, 0, SEEK_END);
    const size_t bytes_in_file = ftell(file_handle);
    fseek(file_handle, 0, SEEK_SET);
    // Read file bytes
    vector<uint8_t> file_data(bytes_in_file);
    fread(file_data.data(), 1, bytes_in_file, file_handle);
    fclose(file_handle);
    CFDataRef file_data_ref = CFDataCreateWithBytesNoCopy(NULL, file_data.data(),
                                                          bytes_in_file,
                                                          kCFAllocatorNull);
    CGDataProviderRef image_provider = CGDataProviderCreateWithCFData(file_data_ref);
    
    // Determine file type, Read image
    NSString *suffix = [file_name pathExtension];
    CGImageRef image;
    if ([suffix isEqualToString: @"png"]) {
        image = CGImageCreateWithPNGDataProvider(image_provider, NULL, true,
                                                 kCGRenderingIntentDefault);
    } else if ([suffix isEqualToString: @"jpg"] ||
               [suffix isEqualToString: @"jpeg"]) {
        image = CGImageCreateWithJPEGDataProvider(image_provider, NULL, true,
                                                  kCGRenderingIntentDefault);
    } else {
        CFRelease(image_provider);
        CFRelease(file_data_ref);
        LOG(ERROR) << "Unknown suffix for file" << file_name;
        return 1;
    }
    
    // Get Image width and height
    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    size_t bits_per_component = CGImageGetBitsPerComponent(image);
    size_t bits_per_pixel = CGImageGetBitsPerPixel(image);
    
    LOG(INFO) << "CGImage width:" << width << " height:" << height << " BitsPerComponent:" << bits_per_component << " BitsPerPixel:" << bits_per_pixel;
    
    size_t image_channels = bits_per_pixel/bits_per_component;
    CGColorSpaceRef color_space;
    uint32_t bitmapInfo = 0;
    if (image_channels == 1) {
        color_space = CGColorSpaceCreateDeviceGray();
        bitmapInfo = kCGImageAlphaNone;
    } else if (image_channels == 4) {
        // Remove alpha channel
        color_space = CGColorSpaceCreateDeviceRGB();
        //bitmapInfo = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big;
        bitmapInfo = kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big;
    } else {
        // FIXME: image convert
        LOG(ERROR) << "Image channel:" << image_channels;
        return false;
    }
    
    // Read Image to bitmap
    size_t bytes_per_row = image_channels * width;
    size_t bytes_in_image = bytes_per_row * height;
    vector<uint8_t> result(bytes_in_image);
    CGContextRef context = CGBitmapContextCreate(result.data(), width, height,
                                                 bits_per_component, bytes_per_row, color_space,
                                                 bitmapInfo);
    LOG(INFO) << "bytes_per_row: " << bytes_per_row;
    // Release resources
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CFRelease(image);
    CFRelease(image_provider);
    CFRelease(file_data_ref);
    
    // Convert Bitmap (channels*width*height) to Matrix (width*height*channels)
    // Remove alpha channel
    int input_channels = input_layer->channels();
    LOG(INFO) << "image_channels:" << image_channels << " input_channels:" << input_channels;
    if (input_channels == 3 && image_channels != 4) {
        LOG(ERROR) << "image_channels !=4,  input_channels=3 not match.";
        return false;
    } else if (input_channels == 1 && image_channels != 1) {
        LOG(ERROR) << "image_channels!=1 input_channels=1 not match.";
        return false;
    }
    //int input_width = input_layer->width();
    //int input_height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (size_t h = 0; h < height; h++) {
        for (size_t w = 0; w < width; w++) {
            for (size_t c = 0; c < input_channels; c++) {
                // OpenCV use BGR instead of RGB
                size_t cc = c;
                if (input_channels == 3) {
                    cc = 2 - c;
                }
                // Convert uint8_t to float
                int index = c*width*height + h*width + w;
                int index2= h*width*image_channels + w*image_channels + cc;
                input_data[index] = static_cast<float>(result[index2]);
                if (mean.size() == input_channels) {
                    input_data[index] -= mean[c];
                }
            }
        }
    }
    return true;
}
//////////////////////////////////////////////////////

@end

@implementation Classifier
-(id)init{
    if(self=[super init]) { }
    Caffe::set_mode(Caffe::CPU); //Caffe::GPU
    return self;
}
-(id)initWithModel:(const string&)model
            weight:(const string&)weight  {
    self=[super init];
    net_.reset(new Net<float>(model, TEST));
    net_->CopyTrainedLayersFrom(weight);
    //CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    //CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    return self;
}
-(id)initWithModel:(const string&)model
           weight:(const string&)weight
              mean:(const string&)mean
             label:(const string&)label  {
    self=[self initWithModel:model weight:weight];
    [self SetMean:mean];
    [self loadLabel:label];
    return self;
}
-(id)initWithModel:(const string&)model
            weight:(const string&)weight
              mean:(const string&)mean  {
    self=[self initWithModel:model
                      weight:weight];
    [self SetMean:mean];
    
    return self;
}
-(void) loadLabel:(const string&) label{
    // Load labels_file
    ifstream labels(label.c_str());
    CHECK(labels) << "Unable to open labels file " << label;
    string line;
    while (getline(labels, line))
        labels_.push_back(string(line));
    
    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}
static bool PairCompare(const pair<float, int>& lhs,
                        const pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static vector<int> Argmax(const vector<float>& v, int N) {
    vector<pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(make_pair(v[i], static_cast<int>(i)));
    partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
    
    vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

-(vector<Prediction>) Classify:(const cv::Mat&) img {
    return [self Classify:img N:5];
}
/* Return the top N predictions. */
-(vector<Prediction>) Classify:(const cv::Mat&) img
                             N:(int) N {
    vector<float> output = [self Predict:img];
    N = min<int>(labels_.size(), N);
    vector<int> maxN = Argmax(output, N);
    vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(make_pair(labels_[idx], output[idx]));
    }
    return predictions;
}

/* Load the mean file in binaryproto format. */
-(void) SetMean:(const string&) mean_file {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    
    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";
    
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

-(vector<float>) Predict:(const cv::Mat&) img {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();
    
    vector<cv::Mat> input_channels;
    [self WrapInputLayer: &input_channels];
    [self Preprocess:img input_channels:&input_channels ];
    
    net_->Forward();
    
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    vector<float> result(begin,end);
    [self fillLabelsWithResult:result];
    return result;
}
-(void) fillLabelsWithResult:(vector<float>) result{
    if(labels_.size()<1){
        vector<size_t> idx(result.size());
        iota(idx.begin(),idx.end(),0);
        sort(idx.begin(),idx.end(),[&result](size_t l,size_t r){return result[l]>result[r];});
        for(int i=0;i<idx.size();i++){
            NSNumberFormatter *formatter = [NSNumberFormatter new];
            NSString *label = [formatter stringFromNumber:[NSNumber numberWithLong: idx[i]]];
            labels_.push_back([label UTF8String]);
        }
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
-(void) WrapInputLayer:(vector<cv::Mat>*) input_channels {
    Blob<float>* input_layer = net_->input_blobs()[0];
    
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

-(void) Preprocess:(const cv::Mat&) img
    input_channels:(vector<cv::Mat>*) input_channels {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;
    
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    cv::Mat sample_normalized;
    if(mean_.empty()){
        sample_normalized = sample_float;
    }else{
        cv::subtract(sample_float, mean_, sample_normalized);
    }
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}


@end
