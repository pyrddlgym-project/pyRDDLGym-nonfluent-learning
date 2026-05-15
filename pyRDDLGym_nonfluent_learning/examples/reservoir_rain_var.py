import numpy as np
import pyRDDLGym

from pyRDDLGym_nonfluent_learning.core.data import generate_rollouts, batch_sampler
from pyRDDLGym_nonfluent_learning.core.learning import JaxNonFluentLearner
from pyRDDLGym_nonfluent_learning.core.uncertainty import LaplacePosterior


if __name__ == '__main__':

     # make some data
    policy = lambda obs: {'release': np.random.uniform(0.0, 30., size=(3,))}
    env = pyRDDLGym.make('Reservoir_Continuous', '0', vectorized=True)
    transitions = generate_rollouts(env, policy, episodes=20, max_steps=80)
    data_iterator = batch_sampler(*transitions, batch_size=128)

    # train model
    model_learner = JaxNonFluentLearner(rddl=env.model, 
                                        param_ranges={
                                            'RAIN_VAR': (0., np.inf)
                                        },
                                        batch_size_train=128)
    for cb in model_learner.optimize_generator(
        data_iterator, epochs=20000, print_progress=True,
        guess={'RAIN_VAR': np.random.uniform(0., 10., size=(3,))}
    ):
        if cb['iteration'] % 500 == 0:
            print(cb['param_fluents'])
    
    # posterior approximation
    single_iterator = batch_sampler(*transitions, batch_size=128, max_iters=1)
    posterior = LaplacePosterior()
    posterior.compile(model_learner, single_iterator, cb['params'], beta=5.0)
    posterior.plot('reservoir_rain_var_posterior.png')
    
    