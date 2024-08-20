# Ensembling-LeNet-with-LSTM-for-Time-Series-Predictions
### Introduction
In the realm of stock price prediction, accurately forecasting future values is a complex challenge due to the inherently volatile and dynamic nature of financial markets. Traditional approaches often rely heavily on manual feature engineering, where domain experts painstakingly craft features based on historical data. While this method has its merits, it can be limited by the subjective nature of feature selection and the inability to capture intricate patterns that may be hidden within the data.

This is where Convolutional Neural Networks (CNNs) come into play. Originally designed for image processing tasks, CNNs have proven to be highly effective in automatically extracting meaningful features from raw data. When applied to time series data, such as stock prices, CNNs can identify patterns and trends that might not be evident through traditional methods. 
**By allowing the time series to define the features themselves better (by keeping the output dimension the same),or enable a more flexible and powerful approach to feature engineering, by expanding the output space.**

In this project, we explore the integration of CNNs with LSTM networks to leverage the strengths of both models. The CNN's ability to automatically extract features from time series data is combined with the LSTM's strength in capturing temporal dependencies, providing a robust solution for stock price prediction.

### LeNet Overview 

![image](https://github.com/user-attachments/assets/49b1fbd6-073d-4c94-8ca4-0d55a5d1777f)
#### Convolution Operations and Pooling: Engineering New Features

In Convolutional Neural Networks (CNNs), the convolution operation plays a crucial role in feature engineering by measuring the overlap between a kernel (often called a filter) and the input space. This is done by sliding the kernel across the input, performing element-wise multiplication and summation at each position, and mapping these results to the output space. The process is handled efficiently through loops within CUDA, making it computationally feasible even for large-scale data.

The output dimensions of a convolution operation can be calculated using the following formula:
dim_of_output = (dim_of_input + 2 * padding_factor - filter_dimension) / striding_factor + 1

In this equation, `nh` and `nw` are the output dimensions calculated by the formula, while `nc'` represents the number of filters used. The result is a feature map that reflects a transformed space, with the filter's learnable parameters being adapted by the network to minimize loss. This effectively means that the convolution operation is not just transforming the data but also discovering and engineering new features that are crucial for reducing prediction error.

Pooling is another fundamental operation, where a specified filter size from the input is reduced by taking the average or maximum value, depending on the type of pooling used. Unlike convolution, pooling keeps the number of channels (`nc'`) the same but reduces the spatial dimensions (`nh` and `nw`), unless padding is applied. This operation helps the model become more invariant to shifts and distortions in the input, by focusing on the most salient features.

A classic example of these operations is found in the LeNet architecture, developed by Yann LeCun ([source](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf). LeNet demonstrates the power of combining convolution and pooling by progressively shrinking the spatial dimensions (`nh` and `nw`) while increasing the number of channels (`nc'`). This process enables the network to learn increasingly complex features at each layer, which is essential for capturing the underlying patterns in the data.

