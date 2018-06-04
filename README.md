# svm-gpu

Support Vector Machine (SVM) library for [Python](https://www.python.org/)) with GPU.

# Support Vector Machines
[Wikipedia](http://en.wikipedia.org/wiki/Support_vector_machine)  :

>Support vector machines are supervised learning models that analyze data and recognize patterns. 
>A special property is that they simultaneously minimize the empirical classification error and maximize the geometric margin; hence they are also known as maximum margin classifiers.
>[![Wikipedia image](http://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)](http://en.wikipedia.org/wiki/File:Kernel_Machine.png)

# Requirements
* NVIDIA CUDA GPU
⋅⋅⋅Compute Capability of the GPU must be at least 3.0.
* CUDA Toolkit
⋅⋅⋅Supported Versions: 7.0, 7.5, 8.0, 9.0 and 9.1.
⋅⋅⋅If you have multiple versions of CUDA Toolkit installed, CuPy will choose one of the CUDA installations automatically. See Working with Custom CUDA Installation for details.
* Python
⋅⋅⋅Supported Versions: 2.7.6+, 3.4.3+, 3.5.1+ and 3.6.0+.
* CuPy
⋅⋅⋅Supported Versions:  4.0.0+.
* NumPy
⋅⋅⋅Supported Versions: 1.9, 1.10, 1.11, 1.12 and 1.13.
⋅⋅⋅NumPy will be installed automatically during the installation of CuPy.

# Quick start
If you are not familiar with SVM I highly recommend this [guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Here's an example of using [svm-gpu](https://github.com/murtazajafferji/svm-gpu) to approximate the XOR function :

```python
import SVM from svm

# initialize a new predictor
svm = SVM('linear', {}}, None, 100, 'ovr', x=x_train, y=y_train, n_folds=3, num_lambda=10, display_plots=True)
svm.fit(x_train, y_train, use_optimal_lambda=True)

# predict things
svm.predict(x_test)

```

More examples are available [here](https://github.com/murtazajafferji/svm-gpu/tree/master/examples).


# API

## Classifiers

Possible classifiers are:

| Classifier  | Type                   | Params         | Initialization                |
|-------------|------------------------|----------------|-------------------------------|
| C_SVC       | multi-class classifier | `c`            | `= new svm.CSVC(opts)`        |
| NU_SVC      | multi-class classifier | `nu`           | `= new svm.NuSVC(opts)`       |
| ONE_CLASS   | one-class classifier   | `nu`           | `= new svm.OneClassSVM(opts)` |
| EPSILON_SVR | regression             | `c`, `epsilon` | `= new svm.EpsilonSVR(opts)`  |
| NU_SVR      | regression             | `c`, `nu`      | `= new svm.NuSVR(opts)`       |

## Kernels

Possible kernels are:

| Kernel  | Parameters                     |
|---------|--------------------------------|
| linear  | No parameter                   |
| poly    | `degree`                       |
| rbf     | `gamma`                        |
| sigmoid | `alpha`, `beta`                |


## Parameters and options

Possible parameters/options are:  

| Name             | Default value(s)       | Description                                                                                           |
|------------------|------------------------|-------------------------------------------------------------------------------------------------------|
| classification\_strategy | `ovr`                | Used classification strategy                                                                            | 
| kernel                   | `rbf`                  | Used kernel                                                                                           |
| degree                   | `3`                     | For `POLY` kernel. Can be a `Number`                                                                 |
| gamma                    | `.5`     | For `RBF` kernel. Can be a `Number`                                                                                 |
| n\_folds                 | `4`                    | `k` parameter for [k-fold cross validation]( http://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). `k` must be >= 1. If `k===1` then entire dataset is use for both testing and training.  |
| use\_optimal\_lambda     | `false`                  |  Whether to use optimal lambda                                                                      |
| logging                  | `false`                 | Whether to show logs                                                                                 |
| display_plots            | `false`                | Whether to show plots                                                                                 |

The example below shows how to use them:

```javascript
var svm = require('svm-gpu');

var clf = new svm.SVM({
    svmType: 'C_SVC',
    c: [0.03125, 0.125, 0.5, 2, 8], 
    
    // kernels parameters
    kernelType: 'RBF',  
    gamma: [0.03125, 0.125, 0.5, 2, 8],
    
    // training options
    kFold: 4,               
    normalize: true,        
    reduce: true,           
    retainedVariance: 0.99, 
    eps: 1e-3,              
    cacheSize: 200,               
    shrinking : true,     
    probability : false     
});
```

##Training

SVMs can be trained using `svm#train(dataset)` method.

Pseudo code : 
```javascript
var clf = new svm.SVM(options);

clf
.train(dataset)
.progress(function(rate){
    // ...
})
.spread(function(trainedModel, trainingReport){
    // ...
});
```

__Notes__ :  
 * `trainedModel` can be used to restore the predictor later (see [this example](https://github.com/murtazajafferji/svm-gpu/blob/master/examples/save-prediction-model-example.js) for more information).
 * `trainingReport` contains information about predictor's accuracy (such as MSE, precison, recall, fscore, retained variance etc.)

## Prediction
Once trained, you can use the classifier object to predict values for new inputs. You can do so : 
 * Synchronously using `clf#predictSync(inputs)`
 * Asynchronously using `clf#predict(inputs).then(function(predicted){ ... });`

**If you enabled probabilities during initialization**  you can also predict probabilities for each class  : 
 * Synchronously using `clf#predictProbabilitiesSync(inputs)`. 
 * Asynchronously using `clf#predictProbabilities(inputs).then(function(probabilities){ ... })`.

__Note__ : `inputs` must be a 1d array of numbers

## Model evaluation
Once the predictor is trained it can be evaluated against a test set. 

Pseudo code : 
```javascript
var svm = require('svm-gpu');
var clf = new svm.SVM(options);
 
svm.read(trainFile)
.then(function(dataset){
    return clf.train(dataset);
})
.then(function(trainedModel, trainingReport){
     return svm.read(testFile);
})
.then(function(testset){
    return clf.evaluate(testset);
})
.done(function(report){
    console.log(report);
});
 ```
# CLI

[svm-gpu](https://github.com/murtazajafferji/svm-gpu/) comes with a build-in Command Line Interpreter.

To use it you have to install [svm-gpu](https://github.com/murtazajafferji/svm-gpu/) globally using `npm install -g svm-gpu`.

See `$ svm-gpu -h` for complete command line reference.


## help
```shell
$ svm-gpu help [<command>]
```
Display help information about [svm-gpu](https://github.com/murtazajafferji/svm-gpu/) 


## train
```shell
$ svm-gpu train <dataset file> [<where to save the prediction model>] [<options>]
```
Train a new model with given data set

__Note__: use `$ svm-gpu train <dataset file> -i` to set parameters values dynamically.

## evaluate
```shell
$ svm-gpu evaluate <model file> <testset file> [<options>]
```
Evaluate model's accuracy against a test set

# How it work

`svm-gpu` uses the official libsvm C++ library, version 3.20. 

For more information see also : 
 * [libsvm web site](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
 * Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
 * [Wikipedia article about SVM](https://en.wikipedia.org/wiki/Support_vector_machine)
 * [node addons](http://nodejs.org/api/addons.html)

# License
MIT