# OCR_LR_V0.1

This project is supposed to do hand written numbers (0 ~ 9) recognition by using a softmax learning model without any intermediate layers. The data set we used for training and testing is [Mnist](http://yann.lecun.com/exdb/mnist/). Then, a cross entropy loss function is applied to evaluate the difference between the predictions and the correct labels. The loss is minimized by using gradient descent method.

## Getting Started

### Prerequisites

  * [Tensorflow](https://www.tensorflow.org/install/) installed
  * [Python](https://www.python.org/) Version: 3.5 installed
  * [Github](https://desktop.github.com/) installed

### Installing
Download this repository:
```
git clone https://github.com/jb892/OCR_LR.git
```

## Running

```
python ocr_lr_test.py
```


## Result:

![](https://github.com/jb892/OCR_LR/blob/master/mnist.png)

Average accuracy of the trainned network on testing dataset = ~ 92% 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

