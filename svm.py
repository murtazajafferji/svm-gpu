import numpy as np 
import cupy as xp 
import scipy.linalg 
import pandas as pd 
import matplotlib.pyplot as plt
import kernels
import itertools


class SVM():
    """support vector machine"""

    def __init__(self, kernel, kernel_params, lambduh=1, max_iter=1000, classification_strategy='ovr', x=None, y=None, n_folds=3, lambda_vals=None, use_optimal_lambda=False, display_plots=False, logging=False):
        """initialize the classifier"""

        self._kernel = kernel
        self._kernel_params = kernel_params
        self._lambduh = lambduh
        self._max_iter = max_iter
        self._classification_strategy = classification_strategy
        self._y = y
        self._set_x(x)

        self._coef_matrix = []

        self._n_folds = n_folds
        self._display_plots = display_plots
        self._logging = logging
        self._lambda_vals = lambda_vals
        if self._lambda_vals is None:
            self._lambda_vals = [10**i for i in range(-3, 4)]
        self._use_optimal_lambda = use_optimal_lambda

    def fit(self, x=None, y=None, prevent_relabel=False, use_optimal_lambda=False):
        """Trains the kernel support vector machine with the huberized hinge loss"""

        self._set_x(x)
        if y is not None:
            self._y = y
        self._K = self._compute_gram_matrix()
        self._n = len(self._x)
        objective_val_size = int(self._max_iter//10) + (1 if self._max_iter % 10 == 0 else 0)
        self._objective_val_per_iter = xp.zeros(objective_val_size)
        
        if self._classification_strategy == 'ovr':
            iterate_over = xp.asarray(np.unique(xp.asnumpy(self._y)))
        elif self._classification_strategy == 'ovo' or self._classification_strategy == 'binary':
            iterate_over = SVM._get_unique_pairs(self._y)

        if self._use_optimal_lambda or use_optimal_lambda:
            self._lambduh, misclassification_error = self.compute_optimal_lambda()
            print('Misclassification error (train), {}, optimal lambda = {} : {}'.format(self._classification_strategy, self._lambduh, misclassification_error))

        for i in range(len(iterate_over)):
            if self._logging:
                print('Training classifier {} of {}'.format(i + 1, len(iterate_over)))
            
            if self._classification_strategy == 'ovr':
                primary_class = iterate_over[i]
                x_filtered, y_filtered = SVM._filter_data_by_class_ovr(self._x, self._y, primary_class)
            elif self._classification_strategy == 'ovo':
                pair = iterate_over[i]
                x_filtered, y_filtered = SVM.filter_data_by_class_ovo(self._x, self._y, pair, prevent_relabel)
            elif self._classification_strategy == 'binary':
                pair = iterate_over[i]
                self._n = len(self._x)
                self._K = self._compute_gram_matrix()
                if prevent_relabel:
                    self._primary_class = 1
                    self._secondary_class = -1
                else:
                    self._primary_class = pair[0]
                    self._secondary_class = pair[1]
                    self._y = xp.where(self._y == self._primary_class, 1, -1)
                self._coef_matrix, self._objective_val_per_iter, self._misclassification_error_per_iter = self._fast_gradient_descent()
                return

            svm = SVM(self._kernel, self._kernel_params, self._lambduh, self._max_iter, 'binary', display_plots= True)
            
            svm.fit(x_filtered, y_filtered, prevent_relabel=True)
            
            self._coef_matrix.append(svm._coef_matrix)
            self._objective_val_per_iter += svm._objective_val_per_iter * (1/len(iterate_over))
        if self._display_plots:
            self.objective_plot()
        return

    def cross_validation_error(self):
        error_per_lambda = xp.zeros(len(self._lambda_vals))

        for i in range(len(self._lambda_vals)):
            lambduh = self._lambda_vals[i]
            if self._logging:
                print('lambduh = {} ({} of {})'.format(lambduh, i + 1, num_lambda))
            error_per_fold = xp.zeros(self._n_folds)
            for j in range(self._n_folds):
                fold_size = int(self._n/self._n_folds)
                indicies = xp.array(range(0, self._n))
                fold_indicies = ((indicies >= fold_size*j) & (indicies <= fold_size*(j+1)))
                x_train = self._x[fold_indicies == True]
                y_train = self._y[fold_indicies == True]       
                x_test = self._x[fold_indicies == False]
                y_test = self._y[fold_indicies == False]
                y_train = y_train.reshape((len(y_train), 1))
                y_test = y_test.reshape((len(y_test), 1))

                y_train = xp.ravel(y_train)
                y_test = xp.ravel(y_test)
                
                svm = SVM(self._kernel, self._kernel_params, lambduh, self._max_iter, self._classification_strategy)
                svm.fit(x_train, y_train)   
                error_per_fold[j] = svm.compute_misclassification_error(x_test, y_test)

            error_per_lambda[i] = xp.mean(error_per_fold)
            
        return error_per_lambda.tolist()

    def compute_optimal_lambda(self):
        cross_validation_error = self.cross_validation_error()
        if self._display_plots:
            df = pd.DataFrame({'lambda':self._lambda_vals, 'Cross validation error':xp.asnumpy(cross_validation_error)})
            display(df)
            df.plot('lambda', 'Cross validation error', logx=True)
            plt.show()
        return self._lambda_vals[np.nanargmin(cross_validation_error)], np.min(cross_validation_error)

    def compute_misclassification_error(self, x, y): 
        y_pred = self.predict(x)
        return xp.mean(y_pred != y)

    def predict(self, x):
        x = self._standardize(x)
        if self._classification_strategy == 'ovr':
            return self._predict_ovr(x)
        elif self._classification_strategy == 'ovo':
            return self._predict_ovo(x)
        else:
            return self._predict_binary(x)

    def objective_plot(self):
        fig, ax = plt.subplots()
        ax.plot(np.array(range(len(self._objective_val_per_iter)))*10, xp.asnumpy(self._objective_val_per_iter), label='Train', c='red')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('Objective value vs. iteration when lambda=' + str(self._lambduh))
        ax.legend(loc='upper right') 
        plt.show()

    def plot_misclassification_error(self): 
        if self._classification_strategy == 'binary':
            fig, ax = plt.subplots() 
            ax.plot(np.array(range(len(self._misclassification_error_per_iter)))*10, xp.asnumpy(self._misclassification_error_per_iter), label='Train', c='red') 
            plt.xlabel('Iteration') 
            plt.ylabel('Misclassification error') 
            plt.title('Misclassification error vs iteration') 
            ax.legend(loc='upper right') 
            plt.show()
        else:
            print('Plotting misclassification error only available for binary classification.')

    def _set_x(self, x):
        if x is not None:
            self._n = len(x)
            self._x_sd = xp.std(x, axis=0)
            self._x_mean = xp.mean(x, axis=0)
            self._x = self._standardize(x)

    def _standardize(self, x):
        sd = self._x_sd
        mean = self._x_mean
        mean[sd == 0] = 0
        sd[sd == 0] = 1
        return (x - mean) / sd

    @staticmethod
    def filter_data_by_class_ovo(x, y, classes, prevent_relabel=False):
        x = SVM._select_classes(x, y, classes)
        y = SVM._select_classes(y, y, classes)
        if prevent_relabel == False:
            y = xp.where(y == classes[0], 1, -1)
                    
        return x, y

    @staticmethod
    def subset_data(x, y, max_samples):
        if max_samples is None or max_samples > len(x):
            return x, y
        else:
            idx = np.random.choice(np.arange(len(x)), max_samples, replace=False)
            return x[idx], y[idx]
    @staticmethod   
    def subset_data_gpu(x, y, max_samples):
        x, y = subset_data(x, y, max_samples)
        return xp.asarray(x), xp.asarray(y)

    @staticmethod
    def _get_unique_pairs(y):
        return pd.Series(list(itertools.combinations(np.unique(xp.asnumpy(y)),2)))

    @staticmethod
    def _select_classes(x, y, classes):
        if len(classes) == 2:
            return x[(y == classes[0]) | (y == classes[1])]
        else:
            return x[xp.asarray(np.isin(xp.asnumpy(y), classes))]

    @staticmethod
    def _select_classes_ovr(x, y, primary_class):
        positive = x[y == primary_class]
        negative = x[y != primary_class]
        # Get random rows of the same length as the positive matrix
        if len(positive) < len(negative):  
            negative = negative[xp.random.choice(negative.shape[0], len(positive), replace=False)]
        return xp.concatenate((positive, negative), axis=0)

    @staticmethod
    def _filter_data_by_class_ovr(x, y, primary_class):
        x = SVM._select_classes_ovr(x, y, primary_class)
        y = SVM._select_classes_ovr(y, y, primary_class)
        y = xp.where(y == primary_class, 1, -1)
        
        return x, y

    def _compute_gradient(self, alpha):
        """Computes the gradient ∇F(β) of F"""
        K_alpha = xp.dot(self._K, alpha)
        grad = -2 / self._n * xp.sum(self._y[:, xp.newaxis] * self._K * xp.max(xp.stack((xp.zeros_like(self._y), 1 - self._y * K_alpha)), axis=0)[:, xp.newaxis], axis=0) + 2 * self._lambduh * K_alpha    
        return grad

    def _objective(self, alpha):
        K_alpha = xp.dot(self._K, alpha)
        return 1 / self._n * xp.sum(xp.max(xp.stack((xp.zeros_like(self._y), 1 - self._y * K_alpha)), axis=0) ** 2) + self._lambduh * alpha.dot(K_alpha)

    def _backtracking_line_search(self, alpha, eta=1, alphaparam=0.5, betaparam=0.8, max_iter=100): 
        grad_alpha = self._compute_gradient(alpha) 
        norm_grad_alpha = xp.linalg.norm(grad_alpha) 
        found_eta = 0 
        iter = 0 
        while found_eta == 0 and iter < max_iter: 
            if self._objective(alpha - eta * grad_alpha) < self._objective(alpha) - alphaparam * eta * norm_grad_alpha ** 2: 
                found_eta = 1 
            elif iter == max_iter: 
                raise ('Max number of iterations of backtracking line search reached') 
            else: 
                eta *= betaparam 
                iter += 1 
            return eta

    def _compute_gram_matrix(self):
        """Computes, for any set of datapoints x1,...,xn, the kernel matrix K"""
        kernel = self._kernel
        if kernel == 'rbf':
            kernel += '_sklearn'
        gram = kernels.kernel_dict[kernel](self._x, self._x, self._kernel_params)
        return gram

    # 
    def _kernel_eval(self, x, x_train):
        keval = kernels.kernel_dict[self._kernel](x_train, x, self._kernel_params)
        return keval

    def _fast_gradient_descent(self):
        eta_init = self._optimal_eta_init()

        alpha = xp.zeros(self._n)
        theta = xp.zeros(self._n)
        eta = eta_init
        grad_theta = self._compute_gradient(theta)
        objective_val_size = int(self._max_iter//10) + (1 if self._max_iter % 10 == 0 else 0)
        objective_vals = xp.ones(objective_val_size)
        misclassification_error_per_iter = xp.ones(objective_val_size)
        iter = 0
        while iter < self._max_iter:
            eta = self._backtracking_line_search(theta, eta=eta)
            alpha_new = theta - eta * grad_theta
            theta = alpha_new + iter / (iter + 3) * (alpha_new - alpha)
            grad_theta = self._compute_gradient(theta)
            alpha = alpha_new
            iter += 1
            if self._display_plots and iter % 10 == 0:
                objective_vals[int(iter/10)] = self._objective(alpha)
                self._coef_matrix = alpha
                misclassification_error_per_iter[int(iter/10)] = self.compute_misclassification_error(self._x, xp.where(self._y > 0, self._primary_class, self._secondary_class))
            
        return alpha, objective_vals, misclassification_error_per_iter

    def _predict_binary(self, x):      
        return self._prediction_binary(self._coef_matrix, x, self._x, self._primary_class, self._secondary_class)

    def _predict_ovo(self, x):        
        def mode(a):
            counts = xp.bincount(a.astype(xp.int64))
            return xp.argmax(counts)
        pairs = SVM._get_unique_pairs(self._y)
        predictions = xp.zeros((x.shape[0], len(pairs)))
        for i in range(len(pairs)):
            pair = pairs[i]
            alpha = self._coef_matrix[i]
            x_train_filtered, y_train_filtered = SVM.filter_data_by_class_ovo(self._x, self._y, pair)
            y_pred = self._prediction_binary(alpha, x, x_train_filtered, pair[0], pair[1])
            predictions[:,i] = y_pred
        return xp.stack([mode(p) for p in predictions])

    def _predict_ovr(self, x):
        y_unique = xp.asarray(np.unique(xp.asnumpy(self._y)))
        prediction_probabilities = xp.zeros((x.shape[0], len(y_unique)))
        for i in range(len(y_unique)):
            alpha = self._coef_matrix[i]
            x_train_filtered, y_train_filtered = SVM._filter_data_by_class_ovr(self._x, self._y, y_unique[i])
            pred_prob = self._prediction_prob(alpha, x, x_train_filtered)
            prediction_probabilities[:,i] = pred_prob 
        return xp.stack([y_unique[xp.argmax(p)] for p in prediction_probabilities])

    def _prediction_prob(self, alpha, x, x_train):
        return xp.dot(self._kernel_eval(x, x_train).T, alpha)

    def _prediction_binary(self, alpha, x, x_train, class1=1, class2=-1):
        pred_prob = self._prediction_prob(alpha, x, x_train)
        return xp.where(pred_prob > 0, class1, class2)
    
    def _optimal_eta_init(self):
        return 1 / scipy.linalg.eigh(xp.asnumpy(2 / self._n * xp.dot(self._K, self._K) + 2 * self._lambduh * self._K), eigvals=(self._n - 1, self._n - 1), eigvals_only=True)[0]