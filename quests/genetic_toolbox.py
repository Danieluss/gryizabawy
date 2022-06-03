from typing import Tuple

import copy
import numpy as np
from quests.individual import Individual
from quests.world_model import WorldModel


class GeneticToolbox:
    """Class containing definition of operators used in evolutionary algorithms."""
    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model

    def crossover(self, a: Individual, b: Individual) -> Tuple[Individual, Individual]:
        i0 = np.random.randint(len(a.initial_state))+1
        i1 = np.random.randint(len(b.initial_state))+1
        f0 = np.random.randint(len(a.final_state))+1
        f1 = np.random.randint(len(b.final_state))+1
        return (
            self.world_model.fix_individual(Individual(a.initial_state[:i0] + b.initial_state[i1:],
                                                       a.final_state[:f0] + b.final_state[f1:])),
            self.world_model.fix_individual(Individual(b.initial_state[:i1] + a.initial_state[i0:],
                                                       b.final_state[:f1] + a.final_state[f0:]))
        )

    def mutate(self, a: Individual) -> Individual:
        a = copy.deepcopy(a)
        update_initial_state = np.random.rand() < 0.5
        x = np.random.rand()
        remove = x < 2/3
        add = x > 1/3

        if update_initial_state:
            arr = a.initial_state
        else:
            arr = a.final_state
        if remove and (len(arr) > 1 or add):
            idx = np.random.randint(len(arr))
            arr.pop(idx)
        if add:
            predicate = self.world_model._sample_predicate()
            arr.append(predicate)

        return self.world_model.fix_individual(a),

    def evaluate(self, a: Individual) -> float:
        return [self.world_model.evaluate_individual(a)]
