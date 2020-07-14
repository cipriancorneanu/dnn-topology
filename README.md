# DNN-TOPOLOGY

We usually use a deep neural network (DNN) to learn a functional mapping between a set of inputs and a desired set of outputs. The aim of this corpus of work is to study the topology of this functional mapping and derive useful insights about learning properties of the network. We provide this code as a basis for computing topological descriptors of deep neural networks.

## Setup

After cloning the repository follow the steps:

1. Install the required packages

```
pip install -r requirements.txt
```

2. Setup OpenMPI

Next you will have to set up OpenMPI, which DIPHA depends on. If you try to build DIPHA without
having OpenMPI (or other compatible MPI library installed) you might get the
following error: "Could NOT find MPI (missing: MPI_C_FOUND MPI_CXX_FOUND)".

Download latest version of [OpenMPI](https://www.open-mpi.org/software/ompi/v4.0/).

Then build following:

```
gunzip -c openmpi-4.0.4.tar.gz | tar xf -
cd openmpi-4.0.4
./configure --prefix=/usr/local
make all install
```

You might find more useful information on their dedicated [website](https://www.open-mpi.org/faq/?category=building#easy-build).  

3. Clone and build DIPHA.

```
git clone https://github.com/DIPHA/dipha.git
cd dipha
mkdir build
cmake -H. build
make
```

This was tested with python3.7.

### Quick start: Train LeNet on MNIST and compute topology

Edit config.py to suit your needs:
- SAVE_PATH sets where to save intermediary and final results. Some results, for example the checkpoints of the models can occupy significant space. Also check the --save_every optional argument you can pass to main.py which sets the frequency of model saving.  
- NPROC sets the number of CPUs used if you want to use mutiple core processing. It speeds up computation.
- UPPER_DIM sets the upper limit for the betti numbers you will comute. Higher dimensional betti numbers can be VERY computational demanding. Start with UPPER_DIM=2 and increase only if possible.

For training LeNet on MNIST and computing the first and second betti curve do:

```
python main.py --net lenet --dataset mnist --trial 0 --lr 0.0005  --n_epochs_train 50 --epochs_test '1 5 10 20 30 40 50' --graph_type functional --train 1 --build_graph 1
```

It will train for 50 epochs and compute topology for the epochs in --epochs_test.
Once finished, you should find a set of files of the form 'adj_epc<EPC>_trl<TRIAL>_<MAX_EPSILON>.bin.out'
in you <SAVE_PATH>/<NETWORK>_<DATASET>/ directory. They contain the persistent homology
results in a specific format.

For visualising the first betti curve, run:

```
python visualize.py --trial 0 --net lenet --dataset mnist --epochs 0 1 3 5 7 10 30 --dim 1

```

For the second betti curve run:
```
python visualise.py --trial 0 --net lenet --dataset mnist --epochs 0 1 5 10 30 50 --dim 2  
```

This might produce something like this:
![alt text](https://github.com/cipriancorneanu/dnn-topology/blob/master/art/betti.png)


Also notice that average life and midlife are computer for each epoch.
Actual results might slightly differ on each training depending on initialization
and hyper-parameters (learning rate, optimization).


## Background

There are two main applications of topologically describing a deep neural network. They are both documented in two CVPR papers (see links below).  


### Early Stopping
By projecting a DNN into a set of Topological Spaces and computing the Betti numbers, we show have shown that:

1. Learning to generalize in DNN is defined by the creation of 2D and 3D cavities in the topological space representing the correlations of activation of distant nodes of the DNN, and the movement of 1D cavities from higher to lower density.
2. Memorizing (overfitting) is indicated by a regression of these cavities toward higher densities in the topological space.

![alt text](https://github.com/cipriancorneanu/dnn-topology/blob/master/art/overview.png)


### Predicting Generalization Gap
There is a high correlation between the generalization gap of a DNN and its topological description. By deriving simple laws among topological projections of a DNN, one can predict its generalization gap without the need of a test set.

![alt text](https://github.com/cipriancorneanu/dnn-topology/blob/master/art/overview_cvpr2020.png)


## More Information
For more information check:
* [CVPR2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Corneanu_What_Does_It_Mean_to_Learn_in_Deep_Networks_And_CVPR_2019_paper.pdf)
* [CVPR2019 poster](https://cipriancorneanu.github.io/files/corneanu2019what_poster.pdf)
* [CVPR2020 paper (oral)](http://openaccess.thecvf.com/content_CVPR_2020/papers/Corneanu_Computing_the_Testing_Error_Without_a_Testing_Set_CVPR_2020_paper.pdf)
* [CVPR2020 spotligh](https://youtu.be/XuDU--076VA)

## Credit
If you are using this in your research please cite:

*"What Does It Mean to Learn in Deep Networks? And, How Does One Detect Adversarial Attacks?"
CA Corneanu, M Madadi, S Escalera, AM Martinez - Proceedings of the IEEE Conference on Computer and Pattern Recognition, 2019*

*"Computing the Testing Error without a Testing Set." Corneanu, Ciprian A., M Madadi, Sergio Escalera, and Aleix M. Martinez. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.*
