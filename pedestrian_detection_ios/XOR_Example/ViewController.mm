//
//  ViewController.m
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import "ViewController.h"
#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import "opencv2/highgui/ios.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <ctime>
using namespace std;
using namespace cv;
#endif


#define KBYTES_CLEAN_UP 10000 //10 Megabytes Max Storage Otherwise Force Cleanup (For This Example We Will Probably Never Reach It -- But Good Practice).
#define LUAT_STACK_INDEX_FLOAT_TENSORS 4 //Index of Garbage Collection Stack Value

@interface ViewController () {
    // Setup the view
    UIImageView *imageView_;
}
@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    // Initialize Deep Network
    self.t = [Torch new];
    [self.t initialize];
    [self.t runMain:@"main" inFolder:@"xor_lua"];
    [self.t loadFileWithName:@"xor_model.net" inResourceFolder:@"xor_lua" andLoadMethodName:@"loadNeuralNetwork"];
    
    
    
    
    // Initialize ImageView
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
    [self.view addSubview:imageView_];
    
    // Read in the image
    UIImage *image = [UIImage imageNamed:@"test.jpg"];
    if(image != nil){
        imageView_.image = image; // Display the image if it is there....
        imageView_.contentMode = UIViewContentModeScaleAspectFit;     //force aspect ratio
    }
    else cout << "Cannot read in the file" << endl;
    cv::Mat cvImage; UIImageToMat(image, cvImage);
    
    
    cv::Mat im;
    cvtColor(cvImage, im, CV_RGB2GRAY);
    resize(im, im, cv::Size(1280,960));
    
    
    HOGDescriptor hog = HOGDescriptor();
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<cv::Rect> box;
    vector<double> weight;
    
    //std::clock_t start = std::clock();
    hog.detectMultiScale(im, box, weight, -1.0, cv::Size(16,16), cv::Size(32,32), 1.05, 2);
    //cout << "Time: " << (double)(std::clock()-start)/(double) CLOCKS_PER_SEC << endl;
    
    
    /*
    for(int i=0;i<box.size();i++){
        //cout<<box.at(i).x<<","<<box.at(i).y<<","<<box.at(i).width<<","<<box.at(i).height<<" : "<<weight.at(i)<<endl;
        //cv::rectangle(cvImage, box.at(i), cvScalar(255.0, 0.0, 0.0));
        cv::rectangle(cvImage, cv::Rect(box.at(i).x/2, box.at(i).y/2, box.at(i).width/2, box.at(i).height/2), cvScalar(255.0, 0.0, 0.0));
    }
    */
    
    XORClassifyObject *classificationObj = [XORClassifyObject new];
    Mat res, resFloat;
    int x, y, w, h;
    
    
    for(int i=0;i<box.size();i++){
        int x = box.at(i).x/2;
        int y = box.at(i).y/2;
        int w = box.at(i).width/2;
        int h = box.at(i).height/2;
        
        if(x < 0)
            x = 0;
        if(x > cvImage.cols)
            x = cvImage.cols;
        if(y < 0)
            y = 0;
        if(y > cvImage.rows)
            y = cvImage.rows;
        if(x+w > cvImage.cols)
            w = cvImage.cols-x;
        if(y+h > cvImage.rows)
            h = cvImage.rows-y;
        
        //cout<<x<<","<<y<<","<<w<<","<<h<<" : "<<weight.at(i)<<endl;
        Mat ROI = cvImage(cv::Rect(x,y,w,h));
        cv::resize(ROI, res, cv::Size(113,113));
        res.convertTo(resFloat, CV_32FC3);
        float *buf = (float*)resFloat.data;
        classificationObj.im = buf;

        
        float predict;
        
        NSDate *methodStart = [NSDate date];
        
        predict = [self classifyExample:classificationObj inState:[self.t getLuaState]];
        
        NSDate *methodFinish = [NSDate date];
        NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
        cout<<"time: "<<executionTime<<endl;
        
        //cout<<"resnet: "<<predict<<endl;
        
        if(predict<0.1)
            cv::rectangle(cvImage, cv::Rect(x, y, w, h), cvScalar(255.0, 0.0, 0.0));
    }
    
    
    
    imageView_.image = MatToUIImage(cvImage);
}

- (BOOL)isValidFloat:(NSString*)string
{
    NSScanner *scanner = [NSScanner scannerWithString:string];
    [scanner scanFloat:NULL];
    return [scanner isAtEnd];
}


- (CGFloat)classifyExample:(XORClassifyObject *)obj inState:(lua_State *)L
{
    NSInteger garbage_size_kbytes = lua_gc(L, LUA_GCCOUNT, LUAT_STACK_INDEX_FLOAT_TENSORS);
    
    if (garbage_size_kbytes >= KBYTES_CLEAN_UP)
    {
        NSLog(@"LUA -> Cleaning Up Garbage");
        lua_gc(L, LUA_GCCOLLECT, LUAT_STACK_INDEX_FLOAT_TENSORS);
    }
    
    //THFloatStorage *classification_storage = THFloatStorage_newWithSize1(2);
    //THFloatTensor *classification = THFloatTensor_newWithStorage1d(classification_storage, 1, 2, 1);


    int numElements = 3*113*113;
    THFloatStorage *classification_storage =  THFloatStorage_newWithData(obj.im, numElements);
    THFloatTensor* classification = THFloatTensor_newWithStorage1d(classification_storage, 0, numElements, 1);

    //THTensor_fastSet1d(classification, 0, obj.im);
    lua_getglobal(L,"classifyExample");
    luaT_pushudata(L, classification, "torch.FloatTensor");
    
    //p_call -- args, results
    int res = lua_pcall(L, 1, 1, 0);
    if (res != 0)
    {
        NSLog(@"error running function `f': %s",lua_tostring(L, -1));
    }
    
    if (!lua_isnumber(L, -1))
    {
        NSLog(@"function `f' must return a number");
    }
    CGFloat returnValue = lua_tonumber(L, -1);
    lua_pop(L, 1);  /* pop returned value */
    return returnValue;
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}

@end

