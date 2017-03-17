#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface OpenCVCaffeUtil : NSObject

+ (void)setup: (UIImage *)image;
+ (NSArray *)predict: (UIImage *)image;

@end
