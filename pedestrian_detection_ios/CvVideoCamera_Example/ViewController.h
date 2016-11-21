//
//  ViewController.h
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "Torch.h"
#include <Torch/Torch.h>
#import "XORClassifyObject.h"
#import <opencv2/highgui/ios.h>

// Slightly changed things here to employ the CvVideoCameraDelegate
@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    CvVideoCamera *videoCamera; // OpenCV class for accessing the camera
}

// Declare internal property of videoCamera
@property (nonatomic, retain) CvVideoCamera *videoCamera;
@property (nonatomic, strong) Torch *t;
@property (weak, nonatomic) IBOutlet UILabel *answerLabel;
@property (weak, nonatomic) IBOutlet UITextField *valueOneTextfield;
@property (weak, nonatomic) IBOutlet UITextField *valueTwoTextField;

@end

