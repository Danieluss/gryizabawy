import numpy as np
import time
from deap.algorithms import eaSimple
from deap import tools, base
from quests.genetic_toolbox import GeneticToolbox
from quests.individual import Individual

from quests.world_model import WorldModel


class QuestGenerator:
    """Class used for quests generation with an evolutionary algorithm"""

    def __init__(self, population_size: int, epochs: int, tournament_size: int,
                 crossover_probability: float, mutation_probability: float,
                 hall_of_fame_size: int = 10):
        self.population_size = population_size
        self.epochs = epochs
        self.tournament_size = tournament_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.hall_of_fame_size = hall_of_fame_size

    def _get_stats_logger(self) -> tools.Statistics:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("stddev", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        def get_time_from_start():
            t_start = time.time()
            return lambda _: time.time() - t_start

        stats.register("time", get_time_from_start())
        return stats

    def run(self, world_model: WorldModel, genetic_toolbox: GeneticToolbox) -> Individual:
        toolbox = base.Toolbox()
        toolbox.register("mate", genetic_toolbox.crossover)
        toolbox.register("mutate", genetic_toolbox.mutate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        toolbox.register("evaluate", genetic_toolbox.evaluate)

        population = []
        for _ in range(self.population_size):
            ind = world_model.sample_individual()
            population.append(ind)

        # TODO maybe remove individuals that can't have a valid plan generated

        stats = self._get_stats_logger()
        hof = tools.HallOfFame(self.hall_of_fame_size)

        _, log = eaSimple(population, toolbox, self.crossover_probability,
                          self.mutation_probability, self.epochs, stats, hof)

        best_ind = None
        for ind in hof:
            if best_ind is None or best_ind.fitness.values[0] < ind.fitness.values[0]:
                best_ind = ind

        return best_ind, log
