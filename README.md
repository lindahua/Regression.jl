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
ret = Regression.solve(
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
  | ``loss`` |  the loss function, which should be an instance of [UnivariateLoss](http://empiricalrisksjl.readthedocs.org/en/latest/loss.html#loss-functions). |
  | ``X``    |  a matrix of inputs (as columns) |
  | ``y``    |  a vector of corresponding outputs |
  | ``bias`` |  The bias term |

  Let ``d`` be the length of each input.
  When ``bias`` is zero, the parameter ``w`` is a vector of length ``d``, and the prediction is given by ``w'x``.
  When ``bias`` is non-zero, the parameter ``w`` is a vector of length ``d+1``, and the prediction is given by
  ``w[1:d]'x + w[d+1]``.

- **MultivariateRegression**(loss, X, Y, k, bias)

  Construct a *multivariate regression* problem, where the prediction is a vector.

  | params |  descriptions |
  |--------|---------------|
  | ``loss`` | the loss function, which should be an instance of [MultivariateLoss](http://empiricalrisksjl.readthedocs.org/en/latest/loss.html#loss-functions). |
  | ``X``    |  a matrix of inputs (as columns) |
  | ``y``    |  a matrix of corresponding outputs (as columns) |
  | ``k``    |  The length of each prediction output |
  | ``bias`` |  The bias term |

  Let ``d`` be the length of each input.
  When ``bias`` is zero, the parameter ``W`` is a matrix of size ``(k, d)``, and the prediction is given by ``W * x``.
  When ``bias`` is non-zero, the parameter ``W`` is a matrix of size ``(k, d+1)``, and the prediction is given by
  ``W[:, 1:d] * x + W[:,d+1]``.


The package also provides convenience functions to construct common problems:

- **linearreg**(X, Y[; bias=0])

  Construct a linear regression problem.

  When ``Y`` is a vector, it is a univariate regression problem, when ``Y`` is a matrix, it is a multivariate regression problem.

  Note that each column of ``X`` corresponds to a sample. The same applies to ``Y`` when ``Y`` is a matrix.

- **logisticreg**(X, y[; bias=0])

  Construct a logistic regression problem.

- **mlogisticreg**(X, y, k[; bias=0])

  Construct a multinomial logistic regression problem.

  Here, ``X`` is a sample matrix, ``y`` is a vector of class labels (values in ``1:k``), and ``k`` be the number of classes.


### Solving Problems

With a constructed problem, you can solve the problem with the ``solve`` function.

**Note:** The ``solve`` function is not exported (in order to avoid confliction with other optimization packages). You should write ``Regression.solve`` when calling this function.

- **Regression.solve**(pb[; ...])

  Solve the regression problem ``pb``, which can be constructed using the construction functions above.

  This function allows the users to supply the following keyword arguments:

  | params | description |
  |--------|-------------|
  | reg | The regularizer. (See [documentation on regularizers](http://empiricalrisksjl.readthedocs.org/en/latest/regularizers.html#regularizers) for details.) |
  | init | The initial guess of the parameters. (If omitted, we use all-zeros as initial guess by default) |
  | solver | The chosen solver (see below for details). The default is ``BFGS()`` |
  | options | The options to control the solving procedure (see below for details) |
  | callback | The callback function, which will be invoked at each iteration. in the following way: ``callback(t, theta, v, g)``, where ``t`` is the iteration number, ``theta`` is the solution at current step, ``v`` is the current objective value, and ``g`` is the current gradient. Default is ``no_op``, which does nothing. |

- **Regression.Options**(...)  

  Construct an option struct to control the solving procedure.

  It accepts the following keyword arguments:

  | params | description |
  |--------|-------------|
  | maxiter |  The maximum number of iterations (default = `200`) |
  | ftol | Tolerance of function value changes (default = `1.0e-6`) |
  | xtol | Tolerance of solution change (default = `1.0e-8`) |
  | grtol | Tolerance of the gradient norm (default = `1.0e-8`) |
  | armijo | The *Armijo* coefficient in line search |
  | beta |  The back tracking ratio in line search |
  | verbosity | The level of display, which is a symbol, whose value can be ``:none``, ``:final``, or ``:iter``. (default = ``:none``) |


### Solvers

As mentioned, the package implements a variety of solvers, one can construct a solver using the following functions:

```julia
GD()       # Gradient descent
BFGS()     # BFGS Quasi-Newton method
LBFGS(m)   # L-BFGS method (with history size m)
ProxGD()   # Proximal gradient descent (suitable for sparse learning, etc)

# the following solver remains in experimental status
AGD()      # Accelerated gradient descent
ProxAGD()  # Accelerated proximal gradient descent
```

---

## Lower Level Interface

Those who care more on performance can directly call the ``Regression.solve!`` function, as follows:

```julia
# Note: solve! will update the solution θ inplace
function solve!{T<:FloatingPoint}(
    solver::DescentSolver,  # the chosen solver
    f::Functional{T},       # the objective functional
    θ::Array{T},            # the solution (which would be updated inplace)
    options::Options,       # options to control the procedure
    callback::Function)     # callback function

# Here, the functional f can be constructed using the following functions:

# empirical risk minimization
f = RiskFun(rmodel, X, Y)   # rmodel is the risk model

# regularized empirical risk minimization
f = RegRiskFun(rmodel, reg, X, Y)   # rmodel is the risk model, reg is the regularizer
```

---

## Algebraic Solution to Linear & Ridge Regression

Note that for linear regression and ridge regression, there exists analytic solution. The package also provides functions that directly compute the analytic solution to these problems, using linear algebraic methods.

- **llsq**(X, Y; ...)

    Solve a linear least square problem.

    This function allows keyword arguments as follows:

    | params | descriptions |
    |--------|--------------|
    | trans  |  If ``trans == true``, it minimizes ``||X' * theta - Y||^2``, otherwise, if minimizes ``||X * theta - Y ||^2`` (default = ``false``). |
    | bias   |  The bias term, namely the value to be augmented to the inputs. Default = ``0``, which indicates no augmentation |
    | method |  A symbol to indicate the matrix factorization method to be used, whose value can be ``qrlq``, ``orth``, or ``svd``. Default = ``qrlq``. |

- **ridgereg**(X, Y, r; ...)

    Solve a ridge regression problem analytically.

    This function allows keyword arguments as follows:

    | params | descriptions |
    |--------|--------------|
    | trans  |  If ``trans == true``, it minimizes ``||X' * theta - Y||^2 + (r/2) ||theta||^2``, otherwise, if minimizes ``||X * theta - Y ||^2 + (r/2) ||theta||^2`` (default = ``false``). |
    | bias   |  The bias term, namely the value to be augmented to the inputs. Default = ``0``, which indicates no augmentation |
