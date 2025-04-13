# NN From Scratch Implementation with Autograd

This repo implements Neural Network from scratch by building computational (topological) graph for calculating automatic gradient. 

It uses 'Variable' to keep track of the graph similar to pytorch Tensor.

## Usage:
```
python HW2.py
```

next todo:
- 

- write more flexible code to take input lr etc 
- write README for how to use
- report

future work:
- 

- Separate into different folder
- Makes bi directional graph, so softmax can check what is the next loss function
- Add more loss function
- Add more optimizer 
- More layer - Conv, etc

has been implemented:
- 

- test out with the input data
- update parameters
- make analytical gradient test
- input softmax and CCE 
- test out with MSE 
- finish base code  
- test out if loss reduces 