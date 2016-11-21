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

@interface ViewController(){
    UIImageView *imageView_; // Setup the image view
    UITextView *fpsView_; // Display the current FPS
    int64 curr_time_; // Store the current time
}
@end

@implementation ViewController
@synthesize videoCamera;

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    // Initialize Deep Network
    const float eps = 0.00001;
    
    self.t = [Torch new];
    [self.t initialize];
    [self.t runMain:@"main" inFolder:@"xor_lua"];
    [self.t loadFileWithName:@"xor_model.net" inResourceFolder:@"xor_lua" andLoadMethodName:@"loadNeuralNetwork"];
    
    
    
    
    // Initialize ImageView
    float cam_width = 720; float cam_height = 1280;
    
    // Take into account size of camera input
    int view_width = self.view.frame.size.width;
    int view_height = (int)(cam_height*self.view.frame.size.width/cam_width);
    int offset = (self.view.frame.size.height - view_height)/2;
    
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, offset, view_width, view_height)];
    
    //[imageView_ setContentMode:UIViewContentModeScaleAspectFill]; (does not work)
    [self.view addSubview:imageView_]; // Add the view
    
    // Initialize the video camera
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView_];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 5; // Set the frame rate
    self.videoCamera.grayscaleMode = NO; // Get grayscale
    self.videoCamera.rotateVideo = YES; // Rotate video so everything looks correct
    
    // Choose these depending on the camera input chosen
    //self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    //self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1280x720;
    
    // Finally add the FPS text to the view
    fpsView_ = [[UITextView alloc] initWithFrame:CGRectMake(0,15,view_width,std::max(offset,35))];
    [fpsView_ setOpaque:false]; // Set to be Opaque
    [fpsView_ setBackgroundColor:[UIColor clearColor]]; // Set background color to be clear
    [fpsView_ setTextColor:[UIColor redColor]]; // Set text to be RED
    [fpsView_ setFont:[UIFont systemFontOfSize:18]]; // Set the Font size
    [self.view addSubview:fpsView_];
    
    // Finally show the output
    [videoCamera start];
}

- (void) processImage:(cv:: Mat &)cvImage
{
    cv::Mat im;
    cvtColor(cvImage, im, CV_RGBA2GRAY);
    //resize(im, im, cv::Size(1280,960));     //make sure we get the small people
    
    
    HOGDescriptor hog = HOGDescriptor();
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<cv::Rect> box;
    vector<double> weight;
    
    //std::clock_t start = std::clock();
    hog.detectMultiScale(im, box, weight, -0.0, cv::Size(16,16), cv::Size(32,32), 1.05, 2);
    //cout << "Time: " << (double)(std::clock()-start)/(double) CLOCKS_PER_SEC << endl;
    
    
    /*
     for(int i=0;i<box.size();i++){
     //cout<<box.at(i).x<<","<<box.at(i).y<<","<<box.at(i).width<<","<<box.at(i).height<<" : "<<weight.at(i)<<endl;
     //cv::rectangle(cvImage, box.at(i), cvScalar(255.0, 0.0, 0.0));
     cv::rectangle(cvImage, cv::Rect(box.at(i).x/2, box.at(i).y/2, box.at(i).width/2, box.at(i).height/2), cvScalar(255.0, 0.0, 0.0));
     }
     */
    
    XORClassifyObject *classificationObj = [XORClassifyObject new];
    Mat res, fRes;
    std::ostringstream ss;
    
    for(int i=0;i<box.size();i++){
        //for(int i=3;i<4;i++){
        int x = box.at(i).x;
        int y = box.at(i).y;
        int w = box.at(i).width;
        int h = box.at(i).height;
        
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
        //cv::resize(ROI, res, cv::Size(4,4));
        
        //res.convertTo(fRes, CV_32FC1);
        cvtColor(res, fRes, CV_BGRA2RGB);
        
        //float *buf = (float*)fRes.data;
        unsigned char *buf = fRes.data;
        //cout<<(int)buf[0]<<endl<<(int)buf[1]<<endl<<(int)buf[2]<<endl<<(int)buf[3]<<endl<<(int)buf[4]<<endl;
        float fbuf[38307];
        for(int k=0;k<38307;k++)
            fbuf[k] = (float)buf[k];
        
        classificationObj.im = fbuf;
        
        float predict;
        
        NSDate *methodStart = [NSDate date];
        
        predict = [self classifyExample:classificationObj inState:[self.t getLuaState]];
        
        NSDate *methodFinish = [NSDate date];
        NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
        cout<<executionTime<<" second"<<endl;
        
        //cout<<"resnet: "<<predict<<endl;
        
        if(predict > 0.5){
            cv::rectangle(cvImage, cv::Rect(x, y, w, h), cvScalar(0.0, 255.0, 255.0));
            ss << predict;
            string s(ss.str());
            putText(cvImage, s, cv::Point(x,y), 1, 1, cv::Scalar(255.0, 0.0, 255.0));
        }
    }
    
    // Finally estimate the frames per second (FPS)
    int64 next_time = getTickCount(); // Get the next time stamp
    float fps = (float)getTickFrequency()/(next_time - curr_time_); // Estimate the fps
    curr_time_ = next_time; // Update the time
    NSString *fps_NSStr = [NSString stringWithFormat:@"FPS = %2.2f",fps];
    
    // Have to do this so as to communicate with the main thread
    // to update the text display
    dispatch_sync(dispatch_get_main_queue(), ^{
        fpsView_.text = fps_NSStr;
    });
    
    //imageView_.image = MatToUIImage(cvImage);
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

