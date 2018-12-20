# StyleMatch
Style transformation tasks involve a content image and a style image. In recent years, explorations have been made to improve the quality of the generated image by achieving better resolutions. We take this problem from a different perspective. Instead of focusing on the result image, we refine the choice of a style image by finding a best style image for a specific content image. The style of an arbitrary style image may not fit the content image. We use a style image which is most similar to the content image as the best fit. The system produces good output images during the test.

The code uses the style transfer example at Kera. https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
The dataset used is from Painter by Numbers at Kaggle. https://www.kaggle.com/c/painter-by-numbers/data

# Reference
L. A. Gatys, A. S. Ecker, and M. Bethge, "A Neural Algorithm of Artistic Style," arXiv:1508.06576v2. 2015. 
