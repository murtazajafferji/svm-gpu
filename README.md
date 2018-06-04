# svm-gpu

Support Vector Machine (SVM) library for [Python](https://www.python.org/)) with GPU

# Support Vector Machines
[Wikipedia](http://en.wikipedia.org/wiki/Support_vector_machine)  :

>Support vector machines are supervised learning models that analyze data and recognize patterns. 
>A special property is that they simultaneously minimize the empirical classification error and maximize the geometric margin; hence they are also known as maximum margin classifiers.
>[![Wikipedia image](http://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)](http://en.wikipedia.org/wiki/File:Kernel_Machine.png)

# Quick start
If you are not familiar with SVM I highly recommend this [guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf).

Here's an example of using [svm-gpu](https://github.com/murtazajafferji/svm-gpu) to approximate the XOR function :

```python
import SVM from svm
import cupy as xp 
import sklearn.model_selection
import time
from svm import SVM
import kernels

xp.random.seed(0)
x_simulated = xp.concatenate((xp.random.normal(-10,10,(100,60)),
                              xp.random.normal(0,10,(100,60)),
                              xp.random.normal(10,10,(100,60)),
                              xp.random.uniform(-1,1,(100,60))),axis=0)

y_simulated = xp.concatenate((xp.tile(0, 100),
                              xp.tile(1, 100),
                              xp.tile(2, 100),
                              xp.tile(3, 100)),axis=0)

# Divide the data into train, test sets 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_simulated, y_simulated)


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
| lambduh                   | `3`                  | Used lambda                                                                                           |
| max_iter                   | `1000`                  | Used maximum number of iterations                                                                                       |

| classification\_strategy | `ovr`                | Used classification strategy                                                                            | 
| x | `x`                | Used x train                                                                            | 
| y | `y`                | Used y train                                                                            | 
| n\_folds                 | `4`                    | `k` parameter for [k-fold cross validation]( http://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). `k` must be >= 1. If `k===1` then entire dataset is use for both testing and training.  |
| lambda\_vals                | `[.001, .01, .1, 1, 10, 100, 1000]`                    | Used lambda vals for cross validation
| use\_optimal\_lambda     | `false`                  |  Whether to use optimal lambda                                                                      |
| display_plots            | `false`                | Whether to show plots                                                                                 |
| logging                  | `false`                 | Whether to show logs                                                                                 |

## Classification strategies

Possible classification strategies are:

| Classification stratgey  | Description |
|-------------|------------------------|
| `binary`       | binary classifier |
| `ovo`      | one-vs-one classifier |
| `ovr`   | one-vs-rest classifier   |

## Kernels

Possible kernels and kernal params are:

| Kernel  | Kernel params                     |
|---------|--------------------------------|
| `linear`  | `{}`                   |
| `poly`    | `{'degree': <int>}`                       |
| `rbf`     | `{'gamma':<float>}`                        |
| `sigmoid` | `{'alpha':<float>, 'beta':<float>}`                |

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