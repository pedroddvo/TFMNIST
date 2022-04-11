A very basic neural network prediction model using the digit MNIST dataset

![Plotting](img/screenshot.png)

## Preface
I roughly read this [example](https://www.tensorflow.org/tutorials/keras/classification) and modified each step to conform to the digit MNIST dataset.
Notable network modifications are:
* reduction in neuron count in the intermediate layer as we are not dealing with the complexity of clothing items
* reduction in epoch count for equal reasons

## Usage
Simply execute the `main.py` script:
```
python main.py
```

The program saves and hot-loads a model to save time upon the next usage of the script
