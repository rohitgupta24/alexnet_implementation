# alexnet_implementation
All about alexnet



It lead to development of other architectures like vgg16.


ALEXNET : 1000 outputs because this was developed n Imagenet datatbase wwhich has 1000 categories.
          (n+2p-f/s)+1
          convo_maxpooling_convo_maxpooling_convo_convo_convo_maxpooling_ Densenet(Full Connection Layer)_FCL _softmax(as 1000 outputs)

Why alexnet worked better ?
They used Relu in hidden networks, its faster than tanh and sigmoid. 
Also they used standardization. For a channel(like in 13*13),it will standardize all pixels in a channel(Local Response Normalization)
Dropout, prevents overfitting.
Enhanced data by data augmentation,it was ahead of time.