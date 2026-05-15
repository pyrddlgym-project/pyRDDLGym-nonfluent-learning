import os
import numpy as np
import pyRDDLGym

from pyRDDLGym_jax.core.planner import load_config, JaxBackpropPlanner, JaxOfflineController

from pyRDDLGym_nonfluent_learning.core.data import generate_rollouts, batch_sampler
from pyRDDLGym_nonfluent_learning.core.learning import JaxNonFluentLearner


if __name__ == '__main__':

     # make some data
    policy = lambda obs: {'force': np.random.uniform(-10, 0) \
                            if obs['ang-pos'] + 0.5 * obs['ang-vel'] < 0.2 * np.random.normal() \
                            else np.random.uniform(0, 10)}
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_env = pyRDDLGym.make('CartPole_Continuous_gym', '0', vectorized=True)
    transitions = generate_rollouts(data_env, policy, episodes=200, max_steps=200)
    data_iterator = batch_sampler(*transitions, batch_size=64)

    # train model
    train_env = pyRDDLGym.make(os.path.join(file_path, 'domain.rddl'), 
                               os.path.join(file_path, 'instance0.rddl'), vectorized=True)
    model_learner = JaxNonFluentLearner(rddl=train_env.model, 
                                        param_ranges={
                                            'W1': (-np.inf, np.inf),
                                            'b1': (-np.inf, np.inf),
                                            'Wacc': (-np.inf, np.inf),
                                            'bacc': (-np.inf, np.inf),
                                            'Wangacc': (-np.inf, np.inf),
                                            'bangacc': (-np.inf, np.inf),
                                        },
                                        batch_size_train=64,
                                        samples_per_datapoint=2,
                                        optimizer_kwargs={'learning_rate': 0.001})
    for cb in model_learner.optimize_generator(
        data_iterator, epochs=40000, print_progress=True,
        guess={'W1': np.random.normal(-0.01, 0.01, size=(3, 4)),
               'b1': np.zeros((4,)),
               'Wacc': np.random.normal(-0.01, 0.01, size=(4,)),
               'bacc': 0.,
               'Wangacc': np.random.normal(-0.01, 0.01, size=(4,)),
               'bangacc': 0.}
    ):
        pass

    # planning in the trained model
    model = model_learner.learned_model(cb['param_fluents'])
    config_path = os.path.join(file_path, 'config.cfg') 
    planner_args, _, train_args = load_config(config_path)
    planner = JaxBackpropPlanner(model, **planner_args)
    controller = JaxOfflineController(planner, **train_args)

    # evaluation of the plan
    test_env = pyRDDLGym.make('CartPole_Continuous_gym', '0', vectorized=True)
    controller.evaluate(test_env, episodes=1, verbose=True, render=True)
    
    