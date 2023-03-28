# fairaudit

This package enables fairness auditing of arbitrary black-box models
given access to a hold-out set.

For an estimator of the mean of a normally-distributed random vector $y$ with known covariance matrix $\sigma^2 I$ given by

$$
    \hat\mu(y) = \mathcal A ~ \text{argmin} \frac{1}{2} \lVert\mathcal A b - y\rVert_2^2 + r(b)
$$

where $r: \mathbb R^p \to \mathbb R$  is a convex function and
$\mathcal A: \mathbb R^p \to \mathbb R^d$ is a linear operator, this package
provides methods to compute Stein's Unbiased  Risk Estimate of $\hat\mu$:

$$
    SURE(\hat\mu, y) = -n \sigma^2 + \lVert\hat\mu(y) - y\rVert_2^2 + 2 \sigma^2 \nabla \cdot \hat\mu(y).
$$

$SURE(\hat\mu, y)$ is a good estimate of the $\ell_2$ risk of $\hat\mu$, especially
for high dimensional problems.

## Installation

fairaudit can be installed with pip.

To install with pip:

```bash
$ pip install fairaudit
```

## Examples

The easiest way to start using fairaudit may be to read the notebooks:

 * [COMPAS](https://github.com/jjcherian/fairaudit/notebooks/compas.ipynb)
 * [Folktables](https://github.com/jjcherian/fairaudit/notebooks/folktables.ipynb)


<!-- ## Usage

There are three key things in this package:

 * The `SURE` class
 * The `Solver` class and its subclasses `CVXPYSolver`, `FISTASolver`, and `ADMMSolver`
 * The `prox_lib` helper library

### The `SURE` Class

The `SURE` class has the following API:
```python
class SURE:
    def __init__(self, variance: float, solver: Solver): ...

    def compute(self, y: torch.Tensor, divergence_parameters={}) -> float:
        """
        Computes and returns SURE for the estimator computed by the solver
        at the point y.

        Currently, divergence_parameters can contain the key "m" to indicate
        how many samples to use during the divergence estimation (which
        dominates the runtime at high dimensions). The default is for m to be
        102.

        In the future we may switch to A-Hutch++ and may change what options
        the divergence_parameters specifies.
        """

    @property
    def solution(self) -> torch.Tensor:
        """
        Returns solver.solve(y) from the last compute call.
        """

    def runtimes(self) -> TypedDict('Runtimes', solver=float, divergence=float):
        """
        Returns how long it took for the solver to run and how long it took
        the divergence estimator to run during the last compute call.
        """
```


### The `Solver` class

Most uses of the library should use one of the existing `Solver` subclasses.
They have the following APIs:

The three notable `Solver` instances provided by this library have the following
constructors:
```python
class FISTASolver(Solver):
    def __init__(self, A: linops.LinearOperator,
                       prox_R: Callable[[torch.Tensor, float | torch.Tensor], torch.Tensor],
                       x0: torch.Tensor,
                       device=None,
                       lipschitz_iterations=20,
                       lipschitz_vec=None,
                       *, max_iters=5000, eps=1e-3):
        """
        This solver solves problems of the form with a variant on FISTA:
              min. 1/2 ||A b - y||_2^2 + r(b)
        and estimates the mean of y with A b^* where b^* is the optimal b.

        A is a linear operator defined using <https://github.com/cvxgrp/torch_linops>

        prox_R is a differentiable-with-respect-to-its-first-argument function to
            find the optimal point b for a (v, t) pair of
              min. t r(b) + 1/2 ||b - v||_2^2

        x0 is the point where we begin iterations, it must be chosen
            indepentently of y.

        lipschitz_iterations is how many iterations of the power method to use
        to approximate the largest eigenvalue of A^T A

        lipschitz_vec is the vector to start the power method. By default, a
        vector of all 1s is used. If this vector is orthogonal to the largest
        eigenvector of A^T A, this argument is mandatory.

        max_iters, eps control when iterations stop.

        """

class ADMMSolver(Solver):
    def __init__(self, A: linops.LinearOperator,
                       prox_R: Callable[[torch.Tensor, float | torch.Tensor], torch.Tensor],
                       x0: torch.Tensor,
                       device=None,
                       *, max_iters=1000, eps_rel=1e-3, eps_abs=1e-6):
        """
        This solver solves problems of the form with a variant on ADMM:
              min. 1/2 ||A b - y||_2^2 + r(b)
        and estimates the mean of y with A b^* where b^* is the optimal b.

        A is a linear operator defined using <https://github.com/cvxgrp/torch_linops>

        prox_R is a differentiable-with-respect-to-its-first-argument function to
            find the optimal point b for a (v, t) pair of
              min. t r(b) + 1/2 ||b - v||_2^2

        x0 is the point where we begin iterations, it must be chosen
            indepentently of y.

        max_iters, eps_rel, eps_abs control when iterations stop.
        """

class CVXPYSolver(Solver):
    def __init__(self, problem: cp.Problem,
                       y_parameter: cp.Parameter, 
                       variables: list[cp.Variable], 
                       estimate: Callable[[list[torch.Tensor]], torch.Tensor]):
        """
        problem must be a CVXPY problem with a single paremeter, y_parameter,
            and variables y_variable.

        estimate must be function which takes tensors with values for each variable
            and returns the estimate.

        WARNING: This solver has poor performance on large problems, and can
        have undetected poor accuracy on some moderately-sized problems.
        """
```

If you wish to implement, `Solver`, it has has the following API, where `T` is
any type of the implementation's choice:
```python
class Solver:

    def solve(self, y: torch.Tensor) -> T:
        """
        Returns intermediate value used to estimate the mean of the distribution
        y is sampled from.
        """

    def estimate(self, beta: T) -> torch.Tensor: ...
        """
        Given the output of a solve call, returns the estimate of the mean of the
        distribution y was sampled from.
        """
```

Note that for a given instance `s` of a solver class, `s.estimate(s.solve(y))` must
be differentiable via torch's backpropagation.


### The `prox_lib` library
Since `FISTASolver` and `ADMMSolver` both require a proximal operator for the
regularizer we provide some methods here to help construct proximal operators:

There are also many helper methods in `surecr.prox_lib`.

 * `prox_l1_norm(v, t)`: the $\ell_1$ norm's proximal operator.
 * `prox_l2_norm(v, t)`: the $\ell_2$ norm's proximal operator.
 * `make_scaled_prox_nuc_norm(shape: tuple[int, int], t_scale: float)`: generates the proximal operator
    $\text{prox}_{r}: \mathbb R^{\mathtt{shape}} \to \mathbb R^{\mathtt{shape}}$
    of 
    $b \mapsto \mathtt{t_scale} \sum_i \sigma_i(b)$
 * `combine_proxs(shape: list[int], proxs: list)`: if there are two regularizers
    $r_1$, $r_2$ such that the regularizer for the problem is given by
    $r(b, b') = r_1(b) + r_2(b')$, then this function should be called with
    `([dim(b), dim(b')], [prox_r_1, prox_r_2])`.
 * `scale_prox(prox, t_scale)`: takes a proximal operator of $r$, and returns the
    proximal operator of $\mathtt{t_scale} r$. -->


# Citing
If you use this code in a research project, please cite the associated paper. 
<!-- ```
@article{nobel2022tractable,
    title={Tractable evalutaion of {S}tein's {U}nbiased {R}isk {E}stimate with convex regularizers},
    author={Parth Nobel \and Emmanuel Cand\`es \and Stephen Boyd},
    publisher = {arXiv},
    year = {2022},
    note = {arXiv:2211.05947 [math.ST]},
    url = {https://arxiv.org/abs/2211.05947},
}
``` -->