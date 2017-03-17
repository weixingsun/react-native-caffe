#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>
#include <numeric>
#include <caffe/caffe.hpp>
//#include "ImageReader.h"
caffe::Net<float> *_net;

@end
@implementation RNCaffe

RCT_EXPORT_MODULE();

RCT_EXPORT_METHOD(runModel:(NSString *)model
                  weight:(NSString *)weight
                  image:(NSString *)image
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject
                  ) {
	    //NSString * port 	= opts[@"port"];
        NSLog(@"use model: %@, \nweight: %@, \nprocess image: %@", model,weight,image);
        _net = new caffe::Net<float>([model UTF8String], caffe::TEST);
        _net->CopyTrainedLayersFrom([weight UTF8String]);
        caffe::Blob<float> *input_layer = _net->input_blobs()[0];
        std::vector<float> mean;
        if(! ReadImageToBlob(test_file_path, mean, input_layer)) {
            NSString * errmsg = @"ReadImageToBlob failed";
            //return resolve(nil);;
            return reject(@"error", errmsg, nil);
        }
    _net->Forward();
    caffe::Blob<float> *output_layer = _net->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    std::vector<float> result(begin, end);
    std::vector<size_t> result_idx(result.size());
    std::iota(result_idx.begin(), result_idx.end(), 0);
    std::sort(result_idx.begin(), result_idx.end(),
              [&result](size_t l, size_t r){return result[l] > result[r];});
    for (int i=0; i<result_idx.size(); i++) {
        //LOG(INFO) << "result[" << result_idx[i] << "]=" << result[result_idx[i]];
        //[_console insertText:[NSString stringWithFormat:@"result[%lu] \t= %f\n", result_idx[i], result[result_idx[i]] ] ];
    }
    return resolve(@{});
}

// Read a jpg/png image from file to Caffe input_layer.
// Modified on tensorflow ios example,
// URL: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ios_examples/simple/ios_image_load.mm
bool ReadImageToBlob(NSString *file_name,
                     const std::vector<float> &mean,
                     caffe::Blob<float>* input_layer) {
    // Get file size
    FILE* file_handle = fopen([file_name UTF8String], "rb");
    fseek(file_handle, 0, SEEK_END);
    const size_t bytes_in_file = ftell(file_handle);
    fseek(file_handle, 0, SEEK_SET);
    // Read file bytes
    std::vector<uint8_t> file_data(bytes_in_file);
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
    std::vector<uint8_t> result(bytes_in_image);
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
        LOG(ERROR) << "image_channels input_channels not match.";
        return false;
    } else if (input_channels == 1 && image_channels != 1) {
        LOG(ERROR) << "image_channels input_channels not match.";
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
                input_data[c*width*height + h*width + w] = static_cast<float>(result[h*width*image_channels + w*image_channels + cc]);
                if (mean.size() == input_channels) {
                    input_data[c*width*height + h*width + w] -= mean[c];
                }
            }
        }
    }
    return true;
}

@end
