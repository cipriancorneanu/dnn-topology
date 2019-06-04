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
python main.py --net lenet --dataset mnist --trial 0 --lr 0.001  --n_epochs_train  --epochs_test '5' --graph_type functional --homology_type 'persistent' --train 1 --build_graph 1


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
d