# DNN-TOPOLOGY

We usually use a deep neural network (DNN) to learn a functional mapping between a set of inputs and a desired set of outputs. The aim of this corpus of work is to study the topology of this functional mapping and derive useful insights about learning properties of the network. We provide this code as a basis for computing topological desscriptors of deep neural networks. 

There are two main applications of topologically describing a deep neural network. They are both documented in two CVPR papers (see links below).  


## Using Topology for Early Stopping
By projecting a DNN into a set of Topological Spaces and computing the Betti numbers, we show have shown that: 

1. Learning to generalize in DNN is defined by the creation of 2D and 3D cavities in the topological space representing the correlations of activation of distant nodes of the DNN, and the movement of 1D cavities from higher to lower density.
2. Memorizing (overfitting) is indicated by a regression of these cavities toward higher densities in the topological space.
  
![alt text](https://github.com/cipriancorneanu/dnn-topology/blob/master/art/overview.png)


## Using Topology for Predict Generalization Gap
There is a high correlation between the generalization gap of a DNN and its topological description. By deriving simple laws among topological projections of a DNN, one can predict its generalization gap without the need of a test set.

![alt text](https://github.com/cipriancorneanu/dnn-topology/blob/master/art/overview_cvpr2020.png)




### Prerequisites
Make sure you have installed the following:

* torch
* torchvision
* dipha


### Installing

For installing DIPHA check https://github.com/DIPHA/dipha. After cloning the repository, build using:

```
cd ./dipha
cmake -H. .Bbuild
cmake --build build -- -j3	
```

### Quick start: Train LeNet on MNIST and compute topology
Edit scripts/config.py to suit your needs:
- SAVE_PATH sets where to save intermediary and final results. Some results, for example the checkpoints of the models can occupy significant space. Also check the --save_every optional argument you can pass to main.py which sets the frequency of model saving.  
- NPROC sets the number of CPUs used if you want to use mutiple core processing. It speeds up computation. 
- UPPER_DIM sets the upper limit for the betti numbers you will comute. Higher dimensional betti numbers can be VERY computational demanding. Start with UPPER_DIM=2 and increase only if possible. 

For training LeNet on MNIST and computing the first betti curve do:
```
python main.py --net lenet --dataset mnist --trial 0 --lr 0.001  --n_epochs_train 50 --epochs_test '10 20 30 40 50' --graph_type functional --homology_type 'persistent' --train 1 --build_graph 1
```
It will train for 50 epochs and compute topology for the epochs in --epochs_test.

[//]: # (### Support models  Currently LeNet and VGG16 are supported. )

## More Information
For more information check:
* [CVPR2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Corneanu_What_Does_It_Mean_to_Learn_in_Deep_Networks_And_CVPR_2019_paper.pdf)
* [CVPR2019 poster](https://cipriancorneanu.github.io/files/corneanu2019what_poster.pdf)
* [CVPR2020 paper (oral)](http://openaccess.thecvf.com/content_CVPR_2020/papers/Corneanu_Computing_the_Testing_Error_Without_a_Testing_Set_CVPR_2020_paper.pdf)
* [CVPR2020 spotligh](https://cipriancorneanu.github.io/files/5560_oral.mp4)

## Credit 
If you are using this in your research please cite: 

*"What Does It Mean to Learn in Deep Networks? And, How Does One Detect Adversarial Attacks?"
CA Corneanu, M Madadi, S Escalera, AM Martinez - Proceedings of the IEEE Conference on Computer and Pattern Recognition, 2019*

*"Computing the Testing Error without a Testing Set." Corneanu, Ciprian A., M Madadi, Sergio Escalera, and Aleix M. Martinez. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.*
