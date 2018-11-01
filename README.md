## Numpy Implementation of Vanilla RNN for Many-to-One Predictions

This project is a simple implementation of Vanilla RNN in pure numpy for many-to-one problems, such as sentiment analysis using written review or malware identification using system calls.
This project was done with an aim to gain deeper insight and unravel the concepts required in designing and training RNN.

I tested this project on system malware identification dataset.

### Dependencies

1. python-2.7
2. scikit-learn
3. numpy

For python dependencies, run - 

<code>pip install -r requirements.txt</code>
### Instructions

Place data and configure trainModel.py with required hyperparameters.
The hyperparameters are at the beginning of the script.

To start training model, run - 

<code>python ./trainModel.py -modelName path\to\save\model -data path\to\train\data.txt -target path\to\targets.txt</code>

To test model, run -

<code>python ./testModel.py -modelName path\to\model -data path\to\data.txt

