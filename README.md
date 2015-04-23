# Regression

A Julia package for regression analysis.

[![Build Status](https://travis-ci.org/lindahua/Regression.jl.svg?branch=master)](https://travis-ci.org/lindahua/Regression.jl)

---

## Overview

This package is based on [EmpiricalRisks](https://github.com/lindahua/EmpiricalRisks.jl), and provides a set of algorithms to perform regression analysis.

This package supports all regression problems that can be formulated as *regularized empirical risk minimization*, as

![regerm](imgs/regerm.png)

In particular, it supports:

- [x] Linear regression
- [x] Ridge regression
- [x] LASSO
- [x] Logistic regression
- [x] Multinomial Logistic regression
- [x] Problems with customized loss and regularizers

The package also provides a variety of solvers

- [x] Analytical solution (for linear & ridge regression)
- [x] Gradient descent
- [x] BFGS
- [x] L-BFGS
- [x] Proximal gradient descent (recommended for LASSO & sparse regression)
- [x] Accelerated gradient descent (*experimental*)
- [x] Accelerated proximal gradient descent (*experimental*)

---

## High Level Interface

The package provides a high-level interface to simplify typical use.

**Example:**  

The following script shows how one can use this package to perform *logistic regression*:

```julia
d = 3      # sample dimension
n = 1000   # number of samples

# prepare data
w = randn(d+1)    # generate the weight vector
X = randn(d, n)   # generate input features
y = sign(X'w[1:d] + w[d+1] + 0.2 * randn(n))  # generate (noisy) response

# perform estimation
ret = solve(
    logisticreg(X, y; bias=1.0),   # construct a logistic regression problem
    reg=SqrL2Reg(1.0e-2),          # apply squared L2 regularization
    options=Options(verbosity=:iter, grtol=1.0e-6 * n))  # set options

# extract results
w_e = ret.sol
```

The high-level interface involves two parts: *problem construction* and *problem solving*.


### Constructing Problems

The package provide several functions to construct regression problems:

- **UnivariateRegression**(loss, X, Y, bias)

  Construct a *univariate regression* problem, where the both arguments to the loss function are scalars.

  | params |  descriptions |
  |--------|---------------|
  | ``loss`` |  the loss function, which should be an instance of ``UnivariateLoss`` |
  | ``X``    |  an matrix of inputs (as columns) |
  | ``y``    |  a vector of corresponding outputs |
  | ``bias`` |  The bias term |

  Let ``d`` be the length of each input.
  When ``bias`` is zero, the parameter is a vector of length ``d``, and the prediction is given by ``w'x``.
  When ``bias`` is non-zero, the parameter is a vector of length ``d+1``, and the prediction is given by
  ``w[1:d]'x + w[d+1]``.

  
