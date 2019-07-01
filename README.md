# DNN-TOPOLOGY
By projecting a Deep Neural Network into a set of Topological Spaces and computing the Betti numbers, we show that 

1. Learning to generalize in DNN is defined by the creation of 2D and 3D cavities in the topological space representing the correlations of activation of distant nodes of the DNN, and the movement of 1D cavities from higher to lower density.
2. Memorizing (overfitting) is indicated by a regression of these cavities toward higher densities in the topological space.
  
![alt text](https://github.com/cipriancorneanu/dnn-topology/blob/master/art/overview.png)


### Prerequisites
-torch
-torchvision
-dipha


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

[//]: # (
### Support models
Currently LeNet and VGG16 are supported. 
)

## More Information
For more information check:
* [CVPR2019 paper](https://cipriancorneanu.github.io/files/corneanu2019what.pdf)
* [CVPR2019 poster](https://cipriancorneanu.github.io/files/corneanu2019what_poster.pdf)

If you are using this in your research plase cite: 
<em> What Does It Mean to Learn in Deep Networks? And, How Does One Detect Adversarial Attacks?
CA Corneanu, M Madadi, S Escalera, AM Martinez - Proceedings of the IEEE Conference on Computer and Pattern Recognition, 2019 <\em>
