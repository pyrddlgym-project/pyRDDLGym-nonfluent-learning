# pyRDDLGym-nonfluent-learning

Non-fluent learning add-on to pyRDDLGym-jax. Learns specified RDDL non-fluents using gradient descent, implemented in pure JAX.

## Installation

Installation via pip:

```shell
conda install pip
pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-nonfluent-learning
```

## Non-Fluent Learning

Any numerical non-fluents can be learnable with this package, as long as they are reasonably identifiable from the given data set. 

### Dataset Creation 

The first step is to create a dataset. This package provides basic functions for rollouts and batched sampling:

```python
from pyRDDLGym_nonfluent_learning.core.data import generate_rollouts, batch_sampler
transitions = generate_rollouts(env, policy, episodes, max_steps)
data_iterator = batch_sampler(*transitions, batch_size)
```

where env is a gymnasium or pyRDDLGym environment and policy is a callable from state to action dict. The environment should be created with ``vectorized = True```.

### Non-Fluent Learning

To learn specific non-fluents, provide a ``param_ranges`` dict mapping RDDL non-fluent names to tuples of bounds. The optimizer will project the trainable non-fluents to the required ranges during optimization:

```python
from pyRDDLGym_nonfluent_learning.core.learning import JaxNonFluentLearner
model_learner = JaxNonFluentLearner(env.model, param_ranges, batch_size_train)
callback = model_learner.optimize(data_iterator, epochs)
```

Let's extract the final non-fluents and create a RDDL model with the substitution:

```python
param_fluents = callback['param_fluents']
model = model_learner.learned_model(param_fluents)
```

### Uncertainty Quantification

It is possible to do Bayesian uncertainty quantification on the trained non-fluents:

```python
from pyRDDLGym_nonfluent_learning.core.uncertainty import LaplacePosterior
single_iterator = batch_sampler(*transitions, batch_size, max_iters=1)
posterior = LaplacePosterior()
posterior.compile(model_learner, single_iterator, callback['params'])
posterior.plot('posterior_credible_intervals.png')
```

Currently the package provides Laplace approximation and NUTS sampler (requires blackjax).

