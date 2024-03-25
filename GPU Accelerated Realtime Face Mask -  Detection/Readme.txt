Description: - 

Initially, we deployed a CNN model with 98% accuracy, but because we made the model with only 5 convolutional layers when we tried to implement it in real-time, the model did not perform as expected (for example, when we turned the face to the side), As a result, we later adapted MobileNetv2, which is likewise a convolutional neural network with 53 layers. Compared to the CNN model we created initially, we can produce a better real-time by using MobileNetv2.

Must need changes to Execute the Models:

1. Change the path of the dataset directories and pre-trained weights in MobileNetV2.
2. Execute the deployment files ( base_cnn --> deploying-final-cnn, final submission --> detect_Modify_Final ).
3. Make sure the video streaming source was selected accordingly as we have used an external webcam our video streaming source will be 1, if using a default webcam then video streaming should have 0 as it's parameter.

Thank You.
