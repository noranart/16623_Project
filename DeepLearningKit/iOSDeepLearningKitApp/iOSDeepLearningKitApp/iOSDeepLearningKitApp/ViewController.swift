//
//  ViewController.swift
//  iOSDeepLearningKitApp
//
//  Created by Amund Tveit on 13/02/16.
//  Copyright © 2016 DeepLearningKit. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    var deepNetwork: DeepNetwork!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }
    
    override func viewDidAppear(animated: Bool) {
        
        deepNetwork = DeepNetwork()
        
        

        
        
        var randomimage = createFloatNumbersArray(3*224*224)
        for i in 0..<randomimage.count {
            randomimage[i] = Float(arc4random_uniform(255))//255.0
        }
        
        let imageShape:[Float] = [1.0, 3.0, 224.0, 224.0]
        let caching_mode = false
        
        // 2. reset deep network and classify random image
        deepNetwork.loadDeepNetworkFromJSON("student", inputImage: randomimage, inputShape: imageShape, caching_mode:caching_mode)
        deepNetwork.classify(randomimage)
        showCIFARImage(randomimage)
        //exit(0)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    //***********************************************************************************
    
    func showCIFARImage(cifarImageData:[Float]) {
        var cifarImageData = cifarImageData
        let size = CGSize(width: 224, height: 224)
        let rect = CGRect(origin: CGPoint(x: 0,y: 0), size: size)
        
        UIGraphicsBeginImageContextWithOptions(size, false, 0)
        UIColor.whiteColor().setFill() // or custom color
        UIRectFill(rect)
        var image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        // CIFAR 10 images are 32x32 in 3 channels - RGB
        // it is stored as 3 sequences of 32x32 = 1024 numbers in cifarImageData, i.e.
        // red: numbers from position 0 to 1024 (not inclusive)
        // green: numbers from position 1024 to 2048 (not inclusive)
        // blue: numbers from position 2048 to 3072 (not inclusive)
        for i in 0..<32 {
            for j in 0..<32 {
                let r = UInt8(cifarImageData[i*224 + j])
                let g = UInt8(cifarImageData[224*224 + i*224 + j])
                let b = UInt8(cifarImageData[2*224*224 + i*224 + j])
                
                // used to set pixels - RGBA into an UIImage
                // for more info about RGBA check out https://en.wikipedia.org/wiki/RGBA_color_space
                image = image.setPixelColorAtPoint(CGPoint(x: j,y: i), color: UIImage.RawColorType(r,g,b,255))!
                
                // used to read pixels - RGBA from an UIImage
                _ = image.getPixelColorAtLocation(CGPoint(x:i, y:j))
            }
        }
        print(image.size)
        
        // Displaying original image.
        let originalImageView:UIImageView = UIImageView(frame: CGRectMake(20, 20, 10*image.size.width, 10*image.size.height))
        originalImageView.image = image
        self.view.addSubview(originalImageView)
    }
    
    
    
}
