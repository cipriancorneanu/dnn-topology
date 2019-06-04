# DNN-TOPOLOGY

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

torch
torchvision
dipha


### Installing

For installing DIPHA check https://github.com/DIPHA/dipha. After cloning the repository build using:

```
cd ./dipha
cmake -H. .Bbuild
cmake --build build -- -j3	
```

### Usage
```
python main.py --net lenet --dataset mnist --trial 0 --lr 0.001  --n_epochs_train  --epochs_test '5' --graph_type functional --homology_type 'persistent' --train 1 --build_graph 1
```

## Authors

* **Ciprian Corneanu** - *Initial work* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

* [CVPR2019 paper](https://cipriancorneanu.github.io/files/corneanu2019what.pdf)
* [CVPR2019 poster] (https://cipriancorneanu.github.io/files/corneanu2019what_poster.pdf)
