# Neural Network From Scratch Implementation with Autograd

This repo implements Neural Network from scratch by building computational (topological) graph for calculating automatic gradient. 

It uses 'Variable' to keep track of the graph similar to pytorch Tensor.

## Usage:
To show all available options use 
```bash
python HW2.py --help
```

Example CLI output
```bash
usage: HW2.py [-h] [--num_classes NUM_CLASSES] [--batch_size BATCH_SIZE]
              [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
              [--optimizer {sgd,adam}] [--momentum MOMENTUM] [--train TRAIN]
              [--test TEST] [--weight_decay WEIGHT_DECAY] [--model {0,1}]

Training and testing configuration

optional arguments:
  -h, --help            show this help message and exit
  --num_classes NUM_CLASSES
                        Number of Classes (default: 3)
  --batch_size BATCH_SIZE
                        Batch size for training and testing (default: 32)
  --epochs EPOCHS       Number of training epochs (default: 100)
  --learning_rate LEARNING_RATE
                        Learning rate for optimizer (default: 100)
  --optimizer {sgd,adam}
                        Choose the optimizer: sgd, or adam (default: sgd)
  --momentum MOMENTUM   momentum for SGD optimizer (default: 0.0)
  --train TRAIN         Specify folder of train data (default: Data_train)
  --test TEST           Specify folder of test data (default: Data_test)
  --weight_decay WEIGHT_DECAY
                        Specify Regularization/ weight decay term (default:
                        0.0)
  --model {0,1}         Specify which model to use: 0 - 2 layer NN, 1 - 3
                        layer NN (default: 0)

```

Usage with default settings
```bash
python HW2.py
```

Usage with custom settings
```bash
python HW2.py --learning_rate 0.001 --train path/to/train --test path/to/test
```

Usage with custom settings another model
```bash
python HW2.py --learning_rate 0.01 --model 1
```

Format of train and test directories 
```
Data_train
|- class1_name (i.e. Carambula)
|- class2_name (i.e. Lychee)
|- class3_name (i.e. Pear)
```

## Future Work
- Separate into different folder
- Makes bi directional graph, so softmax can check what is the next loss function
- Add more loss function
- Add more optimizer 
- More layer - Conv, etc

## Implemented
- [x] write README for how to use
- [x] report
- [x] write more flexible code to take input lr etc 
- [x] test out with the input data
- [x] update parameters
- [x] make analytical gradient test
- [x] input softmax and CCE 
- [x] test out with MSE 
- [x] finish base code  
- [x] test out if loss reduces 


