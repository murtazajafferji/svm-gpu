# svm-gpu

Multiclass Support Vector Machine (SVM) library for [Python](https://www.python.org/) with GPU. This is a fast and dependable classification algorithm that performs very well with a limited amount of data.

# Support Vector Machines
[Wikipedia](http://en.wikipedia.org/wiki/Support_vector_machine):

>Support vector machines are supervised learning models that analyze data and recognize patterns. 
>A special property is that they simultaneously minimize the empirical classification error and maximize the geometric margin; hence they are also known as maximum margin classifiers.
>[![Wikipedia image](http://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)](http://en.wikipedia.org/wiki/File:Kernel_Machine.png)


The advantages of support vector machines are:  
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.  
* Higher speed and better performance with a limited number of samples (in the thousands) compared to neural networks  

The disadvantages of support vector machines include:  
* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation

Applications: 
* SVMs are helpful in text and hypertext categorization as their application can significantly reduce the need for labeled training instances in both the standard inductive and transductive settings.
* Classification of images can also be performed using SVMs. Experimental results show that SVMs achieve significantly higher search accuracy than traditional query refinement schemes after just three to four rounds of relevance feedback.
* Hand-written characters can be recognized using SVM

# Quick start

Here's an example of using [svm-gpu](https://github.com/murtazajafferji/svm-gpu) to predict labels for images of hand-written digits:

```python
import cupy as xp 
import sklearn.model_selection
from sklearn.datasets import load_digits
from svm import SVM

# Load the digits dataset, made up of 1797 8x8 images of hand-written digits
digits = load_digits()

# Divide the data into train, test sets 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target)

# Move data to GPU
x_train = xp.asarray(x_train)
x_test =  xp.asarray(x_test)
y_train = xp.asarray(y_train)
y_test = xp.asarray(y_test)

# initialize a new predictor
svm = SVM(kernel='rbf', kernel_params={'sigma': 15}, classification_strategy='ovr', x=x_train, y=y_train, n_folds=3, use_optimal_lambda=True, display_plots=True)

svm.fit(x_train, y_train)

# predict things
svm.predict(x_test)

# compute misclassification error
misclassification_error = svm.compute_misclassification_error(x_test, y_test)
print('Misclassification error, lambda = {} : {}\n'.format(svm._lambduh, misclassification_error))

# ******** Output ********
# Misclassification error, lambda = .1 : 0.015555555555555545

```

More examples are available [here](https://github.com/murtazajafferji/svm-gpu/blob/master/demo.ipynb).


# API

## Parameters and options

Possible parameters/options are: 

| Name             | Default value(s)       | Description                                                                                           |
|------------------|------------------------|-------------------------------------------------------------------------------------------------------|
| kernel                   | Required                  | Used kernel                                                                                           |
| kernel\_params                   | Required                  | Used k parameters                                                                                           |
| lambduh                   | `1`                  | Used lambda                                                                                           |
| max\_iter                   | `1000`                  | Used maximum number of iterations     |
| classification\_strategy | `ovr`                | Used classification strategy                | 
| x | `None`                | Used x train                                                                            | 
| y | `None`                | Used y train                                                                            | 
| n\_folds                 | `3`                    | `k` parameter for [k-fold cross validation]( http://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). `k` must be >= 1. If `k===1` then entire dataset is use for both testing and training.  |
| lambda\_vals                | `[.001, .01, .1, 1, 10, 100, 1000]`                    | Used lambda vals for cross validation
| use\_optimal\_lambda     | `false`                  |  Whether to use optimal lambda                                                                      |
| display_plots            | `false`                | Whether to show plots                                                                                 |
| logging                  | `false`                 | Whether to show logs                                                                                 |

## Classification strategies

Possible classification strategies are:

| Classification stratgey  | Description |
|-------------|------------------------|
| `binary`       | binary classification |
| `ovo`      | one-vs-one classification |
| `ovr`   | one-vs-rest classification   |

## Kernels

Possible kernels and kernal params are:

| Kernel  | Kernel params                             | Notes                           |
|---------|-------------------------------------------|-------------------------------- |
| `linear`  | `{}`                                    | Use when number of features is larger than number of observations. |
| `poly`    | `{'degree': <int>}`                     | |
| `rbf`     | `{'gamma':<float>}`                     | Use Gaussian Radial Basis Function (rbf) kernel when number of observations is larger than number of features. If number of observations is larger than 50,000 speed could be an issue when using gaussian kernel; hence, one might want to use linear kernel.                  |
| `sigmoid` | `{'alpha':<float>, 'beta':<float>}`     | |

# Requirements
* NVIDIA CUDA GPU  
  Compute Capability of the GPU must be at least 3.0.
* CUDA Toolkit  
  Supported Versions: 7.0, 7.5, 8.0, 9.0 and 9.1.  
  If you have multiple versions of CUDA Toolkit installed, CuPy will choose one of the CUDA installations automatically. See Working with Custom CUDA Installation for details.
* Python  
  Supported Versions: 2.7.6+, 3.4.3+, 3.5.1+ and 3.6.0+.
* CuPy  
  Supported Versions:  4.0.0+.
* NumPy  
  Supported Versions: 1.9, 1.10, 1.11, 1.12 and 1.13.  
  NumPy will be installed automatically during the installation of CuPy.

# License
MIT