# March Maddness

This is a repository documenting my yearly quest to predict the NCAA Mens Basketball Tournament.

## What Is Not Included

All collected data and trained models are kept locally on my machine. Any data should be collected on your own.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
Uses 64 bit python. See [requirements.txt](requirements.txt) for required packages.

### Example
#### Predict:
```sh
python main.py predict -y 2026 -f "folder\trained_model.npy" -l [NN.Sigmoid(38,64),NN.Sigmoid(64,1)]
```

#### Train:
```sh
python main.py train -e 500 -f "folder\model_to_train" -l [NN.Sigmoid(38,64),NN.Sigmoid(64,1)]
```
```sh
python main.py train -e 500 -f "folder\model_to_load.npy" -l [NN.Sigmoid(38,64),NN.Sigmoid(64,1)] --load
```

## License

This project is licensed under the MIT License.

## Results
> | year | first round correct | overall points |
> |-|-|-|
> | 2025 |25 | 490 |

## Updates

### 2025

- Initial implementation

### 2026

- Added Bracket class
- cleaned up and prepared repo for github.
- Fixed bug in Sigmoid.
- Added CLI
