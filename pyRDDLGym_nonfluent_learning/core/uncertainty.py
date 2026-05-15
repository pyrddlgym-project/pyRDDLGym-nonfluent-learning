from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp
from jax import random

from pyRDDLGym_jax.core.compiler import JaxRDDLSimState

from pyRDDLGym_nonfluent_learning.core.learning import JaxNonFluentLearner

State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
Transition = Tuple[State, Action, State]
DataStream = Iterable[Transition]
Params = Dict[str, np.ndarray]
PosteriorResult = Dict[str, Any]


class Posterior(ABC):
    '''Abstract base class for posterior inference methods over non-fluent parameters.'''

    @abstractmethod
    def compile(self, learner: JaxNonFluentLearner, data: DataStream, 
                map_params: Params, **kwargs) -> PosteriorResult:
        pass

    def sample(self, key: random.PRNGKey, num_samples: int) -> State:
        '''Draws samples from the approximated posterior distribution.
        
        :param key: JAX PRNG key for sampling
        :param num_samples: number of posterior samples to draw
        '''
        posterior = getattr(self, 'posterior', None)
        if posterior is None:
            raise ValueError("Posterior has not been compiled yet.")
        samples = posterior['sample_fn'](key, num_samples)
        return samples

    def plot(self, save_name: str) -> None:
        '''Plots the posterior summary with mean and credible intervals per parameter.
        
        :param save_name: filename to save the plot (e.g. 'posterior.png')
        '''
        posterior = getattr(self, 'posterior', None)
        learner = getattr(self, 'learner', None)
        if posterior is None or learner is None:
            raise ValueError("Posterior has not been compiled yet.")
        
        # extract mean and 95% credible intervals per parameter
        ci95 = posterior['credible_interval']
        labels = []
        rows = []
        for name in learner.param_ranges:
            mean_value = np.asarray(posterior['map'][name]).reshape(-1)
            lower_value = np.asarray(ci95['lower'][name]).reshape(-1)
            upper_value = np.asarray(ci95['upper'][name]).reshape(-1)
            for idx, (mean_i, lower_i, upper_i) in enumerate(
                zip(mean_value, lower_value, upper_value)
            ):
                label = name if mean_value.size == 1 else f'{name}[{idx}]'
                rows.append((label, mean_i, lower_i, upper_i))

        # sort parameters by mean value for better visualization
        rows.sort(key=lambda item: item[2], reverse=True)
        labels, means, lowers, uppers = zip(*rows)
        ypos = np.arange(len(rows))
        widths = np.asarray(uppers) - np.asarray(lowers)
        fig_h = max(4.0, 0.35 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(10.0, fig_h))
        ax.errorbar(
            means,
            ypos,
            xerr=[np.asarray(means) - np.asarray(lowers), 
                  np.asarray(uppers) - np.asarray(means)],
            fmt='o',
            color='#1f77b4',
            ecolor='#1f77b4',
            elinewidth=1.5,
            capsize=3,
            markersize=4,
        )
        ax.scatter(means, ypos, c=widths, cmap='viridis', s=24, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Posterior mean with 95% interval')
        ax.set_title('Laplace posterior summary over non-fluents')
        ax.grid(axis='x', alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)


class LaplacePosterior(Posterior):
    '''Approximates the posterior around MAP using a full-matrix Laplace method.'''

    def compile(self, learner: JaxNonFluentLearner, 
                data: DataStream,
                map_params: Params,
                key: Optional[random.PRNGKey]=None,
                beta: float=1.0,
                prior_precision: float=1e-6,
                damping: float=1e-6) -> PosteriorResult:
        '''Approximates the posterior around MAP using a full-matrix Laplace method.
        The objective is the generalized Bayes energy beta * sum_i L_i(theta) with
        an isotropic Gaussian prior centered at MAP.

        :param learner: the trained JaxNonFluentLearner instance
        :param data: sequence of transition batches
        :param map_params: MAP parameters from training (unmapped parameter space)
        :param key: JAX PRNG key (derived from clock if not provided)
        :param beta: inverse temperature scaling the loss contribution
        :param prior_precision: isotropic Gaussian prior precision at MAP
        :param damping: additional diagonal stabilization for numerical robustness
        '''
        self.learner = learner
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))

        # stack all batches into (num_batches, batch_size_train, ...) arrays for lax.scan
        batches = list(data)
        if not batches:
            raise ValueError('infer_posterior_laplace requires at least one batch.')
        key, scan_key = random.split(key)
        batch_keys = random.split(scan_key, len(batches))

        # initialize simulation state and flatten parameters
        fls, nfls = learner._batched_init_subs()
        hyperparams = learner.compiled.model_aux['params']
        sim_state = JaxRDDLSimState(key=key, fls=fls, nfls=nfls, model_params=hyperparams)
        flat_map, unravel_fn = jax.flatten_util.ravel_pytree(map_params)
        
        # per-batch data objective at MAP with batch-specific stochastic key
        def single_batch_data_loss(theta_flat, state_key, batch_size, 
                                   states, actions, next_states):
            params = unravel_fn(theta_flat)
            state_i = sim_state.replace(key=state_key)
            loss = learner.loss_fn(
                state_i, params, states, actions, next_states, learner.norm_stats)
            return beta * loss * batch_size

        grad_fn = jax.jit(jax.grad(single_batch_data_loss, argnums=0))
        hess_fn = jax.jit(jax.hessian(single_batch_data_loss, argnums=0))

        dim = flat_map.size
        grads = []
        hess_accum = np.zeros((dim, dim), dtype=np.float64)
        for i, (states, actions, next_states) in enumerate(batches):
            n_batch = float(np.asarray(states[learner.state_keys[0]]).shape[0])
            grad_i = grad_fn(flat_map, batch_keys[i], n_batch, states, actions, next_states)
            hess_i = hess_fn(flat_map, batch_keys[i], n_batch, states, actions, next_states)
            grad_i = np.asarray(grad_i)
            hess_i = np.asarray(hess_i)
            grads.append(grad_i)
            hess_accum += hess_i
        
        # compute the "outer product" of the gradients for the sandwich covariance
        grads = np.stack(grads, axis=0)
        grads_centered = grads - np.mean(grads, axis=0, keepdims=True)
        score_outer = grads_centered.T @ grads_centered
        
        # prior curvature is added once at MAP (mean centered at map_params)
        hess_accum += prior_precision * np.eye(dim, dtype=hess_accum.dtype)
        hess_accum = 0.5 * (hess_accum + hess_accum.T)

        # eigendecomposition for the covariance and precision matrices
        evals, evecs = np.linalg.eigh(hess_accum)
        evals = np.maximum(evals, damping)
        covariance = (evecs / evals) @ evecs.T

        # sandwich covariance
        cov_sandwich = covariance @ score_outer @ covariance
        cov_sandwich = 0.5 * (cov_sandwich + cov_sandwich.T)
        var_sandwich = np.maximum(np.diag(cov_sandwich), damping)
        std_sandwich = np.sqrt(var_sandwich)
        prec_sandwich_diag = 1.0 / var_sandwich
        cov_eigs = 1.0 / evals
        cov_cond = float(np.max(cov_eigs) / np.maximum(np.min(cov_eigs), damping))
        ci95 = {
            'lower': unravel_fn(jnp.asarray(flat_map - 1.96 * std_sandwich)),
            'upper': unravel_fn(jnp.asarray(flat_map + 1.96 * std_sandwich))
        }

        # sample function
        def sample_fn(rng_key, num_samples):
            samples_flat = random.multivariate_normal(
                rng_key, jnp.asarray(flat_map), jnp.asarray(cov_sandwich), (num_samples,))
            return jax.vmap(unravel_fn)(samples_flat)

        self.posterior = {
            'map': map_params,
            'cov': cov_sandwich,
            'cov_eigs': jnp.asarray(cov_eigs),
            'cov_cond': cov_cond,
            'diag_std': unravel_fn(jnp.asarray(std_sandwich)),
            'diag_prec': unravel_fn(jnp.asarray(prec_sandwich_diag)),
            'credible_interval': ci95,
            'unravel_fn': unravel_fn,
            'sample_fn': jax.jit(sample_fn, static_argnums=(1,)),
        }
        return self.posterior
    

class NUTSPosterior(Posterior):

    def compile(self, learner: JaxNonFluentLearner, 
                data: DataStream,
                map_params: Params,
                key: Optional[random.PRNGKey]=None,
                num_samples: int=500,
                num_warmup: int=200,
                beta: float=1.0,
                prior_precision: float=1e-6) -> PosteriorResult:
        '''Samples the posterior with BlackJAX NUTS and summarizes the draws.

        :param learner: the trained JaxNonFluentLearner instance
        :param data: sequence of transition batches
        :param map_params: MAP parameters from training (unmapped parameter space)
        :param key: JAX PRNG key (derived from clock if not provided)
        :param num_samples: number of posterior samples to draw after warmup
        :param num_warmup: number of warmup steps for adaptation
        :param beta: inverse temperature scaling the loss contribution
        :param prior_precision: isotropic Gaussian prior precision at MAP
        '''
        self.learner = learner
        num_samples = int(num_samples)
        num_warmup = int(num_warmup)
        if key is None:
            key = random.PRNGKey(round(time.time() * 1000))

        # stack all batches into (num_batches, batch_size_train, ...) arrays for lax.scan
        batches = list(data)
        if not batches:
            raise ValueError('compile_posterior requires at least one batch.')
        key, warmup_key, data_key = random.split(key, 3)
        batch_keys = list(random.split(data_key, len(batches)))

        # initialize simulation state and flatten parameters 
        fls, nfls = learner._batched_init_subs()
        hyperparams = learner.compiled.model_aux['params']
        sim_state = JaxRDDLSimState(key=key, fls=fls, nfls=nfls, model_params=hyperparams)
        flat_map, unravel_fn = jax.flatten_util.ravel_pytree(map_params)

        # per-batch data objective at MAP with batch-specific stochastic key
        def single_batch_data_loss(theta_flat, state_key, batch_size, 
                                   states, actions, next_states):
            params = unravel_fn(theta_flat)
            state_i = sim_state.replace(key=state_key)
            loss = learner.loss_fn(
                state_i, params, states, actions, next_states, learner.norm_stats)
            return beta * loss * batch_size

        @jax.jit
        def logdensity(theta_flat):
            params = unravel_fn(theta_flat)
            in_bounds = jnp.array(True)
            for (name, (lo, hi)) in learner.param_ranges.items():
                value = params[name]
                if lo is not None and np.isfinite(lo):
                    in_bounds = jnp.logical_and(in_bounds, jnp.all(value >= lo))
                if hi is not None and np.isfinite(hi):
                    in_bounds = jnp.logical_and(in_bounds, jnp.all(value <= hi))
            total_loss = 0.0
            for state_key, (states, actions, next_states) in zip(batch_keys, batches):
                batch_size = float(np.asarray(states[learner.state_keys[0]]).shape[0])
                total_loss = total_loss + single_batch_data_loss(
                    theta_flat, state_key, batch_size, states, actions, next_states)
            total_loss = total_loss + 0.5 * prior_precision * jnp.dot(
                theta_flat - flat_map, theta_flat - flat_map)
            return jnp.where(in_bounds, -total_loss, -jnp.inf)

        # adapt the NUTS sampler with BlackJAX
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        (state, parameters), _ = warmup.run(warmup_key, flat_map, num_steps=num_warmup)
        nuts = blackjax.nuts(
            logdensity,
            step_size=parameters['step_size'],
            inverse_mass_matrix=parameters['inverse_mass_matrix']
        )

        # compile step function
        def sample_fn(rng_key, _num_samples):
            def nuts_step(_state, _rng_key):
                _state, _ = nuts.step(_rng_key, _state)
                y = _state.position
                return _state, y
            keys = random.split(rng_key, _num_samples)
            _, samples = jax.lax.scan(nuts_step, state, keys)
            return samples
        sample_fn = jax.jit(sample_fn, static_argnums=(1,))

        # sample from the posterior with the adapted NUTS sampler
        samples = sample_fn(data_key, num_samples)
        
        # summarize the posterior samples
        sample_mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples, rowvar=False, bias=False)
        sample_cov = np.atleast_2d(sample_cov)
        sample_cov = 0.5 * (sample_cov + sample_cov.T)
        sample_var = np.maximum(np.diag(sample_cov), prior_precision)
        sample_std = np.sqrt(sample_var)
        sample_prec_diag = 1.0 / sample_var
        sample_eigs = np.linalg.eigvalsh(sample_cov)
        sample_cond = float(
            np.max(sample_eigs) / np.maximum(np.min(sample_eigs), prior_precision))
        ci_bounds = np.quantile(samples.astype(np.float64), [0.025, 0.975], axis=0)
        lower_flat = np.minimum(ci_bounds[0], ci_bounds[1])
        upper_flat = np.maximum(ci_bounds[0], ci_bounds[1])
        
        self.posterior = {
            'map': map_params,
            'mean': unravel_fn(jnp.asarray(sample_mean)),
            'cov': sample_cov,
            'cov_eigs': jnp.asarray(sample_eigs),
            'cov_cond': sample_cond,
            'diag_std': unravel_fn(jnp.asarray(sample_std)),
            'diag_prec': unravel_fn(jnp.asarray(sample_prec_diag)),
            'credible_interval': {
                'lower': unravel_fn(jnp.asarray(lower_flat)),
                'upper': unravel_fn(jnp.asarray(upper_flat))
            },
            'unravel_fn': unravel_fn,
            'sample_fn': sample_fn,
            'nuts_state': state,
            'nuts_params': parameters,
            'samples': samples,
            'num_samples': num_samples,
            'warmup_steps': num_warmup,
        }
        return self.posterior
