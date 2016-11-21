#Deep Compression for Pedestrian Detection in iOS
Noranart Vesdapunt (nvesdapu)

##Summary:
An attempt to port pedestrian detection model into iOS. The model will be compressed by Knowledge Distillation [1] from ResNet [2] to small manually-designed network. The deep learning framework is based on torch-ios [4].

##Installation:
For Deep Learning Kit, please convert /models/student_iter_42000.caffemodel	to student.json by https://github.com/DeepLearningKit/caffemodel2json (the file size will be 10x).

For pedestrian_detection_ios/mac, please add opencv framework and torch framework to the project directory. Torch framework can be built by /torch-ios/generate_ios_framework.

##Background:
Deep learning in iOS is still an ongoing research area due to the large and slow model. However, with Knowledge Distillation technique, it is possible to train a smaller Student model to mimic a large Teacher model. The idea is to extract soft target from Teacher model to provide as a guidance. These soft targets will act as extra labels and improve Student model performance. Moreover, Student model will be trained with both soft and hard target (ground truth), which provides a chance to correct Teacher mistakes. 

To fully utilize the deep network, I chose to tackle object detection problem, and focus on the pedestrian object. Pedestrian detection is a well-researched topic that have both state-of-the-art models and dataset available. My model will be trained on Caltech Pedestrian Dataset [3] as Pedestrian classification because feeding in the whole image for detection is impossible for iPad memory. I will use built in OpenCv HOG-SVM pedestrian detector as region proposal and pass the bounding boxes to deep network for a better classifier.

##Challenges:
ResNet is obviously too large to even run on iPad. Several researchers attempt to compress these kind of large but powerful networks, and it is still considered as an unsolved problem. Since most of the state-of-the-art techniques in Computer Vision utilizes deep learning, being able to compress it into small device such as iPad surely impact several real world problems.

On the implementation part, torch-ios is an outdated framework. Since this a single person project in only 1 month, if the amount of modifications is out of hand, I might change to Deep Learning Kit [3]. But still, it will require model conversion from Torch to Caffe, which is currently unstable.

##Goals & Deliverables
**Plan To Archive:** A small and reasonable pedestrian detection model for iPad, in term of speed, memory, and performance (log-average miss rate).

**Hope To Archive:** A full fledge working framework and model on iPad

**Success Metric:** The model will be evaluate on Caltech metric, which will reflect the log-average miss rate on Caltech Pedestrian Dataset. Student model should be able to massively compress Teacher model in term of file size and number of parameters, while retaining the performance. I will provide a sample of detection video, and a speed benchmark on Titan X GPU, and if possible iPad Air 2.

**How Realistic:** The model should definitely be finished in 1 month, but whether I could make the entire framework running on iPad is still unknown.

##Schedule
**Nov 6:** Finished project proposal

**Nov 13:** Literature review on model compression. Attempt to build torch-ios

**Nov 16:** Explore Deep Learning Kit	as alternative choice, create iOS project with HOGSVM as pedestrian detection proposal

**Nov 20:** Start training Student model.

**Nov 23:** Attempt to fix crashed layers in torch-ios

**Nov 27:** Reconcile outputs of torch-ios, and current torch7

**Nov 30:** Attempt to port the project to iPad

**Dec 3:** Finalize the project, gather results/issues for report

**Dec 8:** Finished presentation, finish report

##Reference
[1] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[2] K. He, X. Zhang, S. Ren, and J. Sun.  Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.

[3] Amund Tveit, Torbjørn Morland, and Thomas Brox Rrøst. DeepLearningKit - an GPU Optimized Deep Learning Framework for Apple's iOS, OS X and tvOS developed in Metal and Swift, arXiv preprint arXiv:1605.04614v1, 2016.

[4] Clement Farabet, Soumith Chintala, et al. torch-ios, GitHub repository, https://github.com/clementfarabet/torch-ios, 2016.

[5] P. Dollár, C. Wojek, B. Schiele and P. Perona Pedestrian Detection: An Evaluation of the State of the Art PAMI, 2012.
