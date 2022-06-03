from typing import List, Tuple
import numpy as np

from quests.utils.fitness import MaxFitness


class Individual:
    """Class that keeps the initial and final state used to generate a plot.
    """

    def __init__(self, initial_state: List[Tuple[str]], final_state: List[Tuple[str]]) -> None:
        self.initial_state = initial_state
        self.final_state = final_state
        self.fitness = MaxFitness()
