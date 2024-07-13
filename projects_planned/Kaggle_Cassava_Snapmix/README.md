# Kaggle Cassava Competition 2020/2021
 
Transfer-learning seemed to be the obvious solution to this, my first, Kaggle competition, but VGG16 did not do well in my experiments. 

Pretty soon into the competition I stumbled over a notebook by Sachin Prabhu, using the SnapMix algorithm described in 

https://arxiv.org/abs/2012.04846: "SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data". 

There is also a code repo on github:  https://github.com/Shaoli-Huang/SnapMix . 

The code by S.H. is written in PyTorch, I don't know PyTorch but found it interessting to try to implement the very same algorithm directly form the description in the research papers using tf.keras - knowing about Keras from the book by F. Chollet.

My implementation is not doing too well though: LB/PB about 44% while the notebook by S.P. reaches > 80% accuracy , i.e. clearly I must have some mistakes in the code.

 ...so the next project is learning Pytorch and then comming back to SnapMix and give it a new try, be it using TensorFlow, tf.keras or Pytorch.

The ResNet implementation used can be found here: https://www.kaggle.com/xhlulu/tf-keras-resnet?select=resnet50_notop.h5
 
