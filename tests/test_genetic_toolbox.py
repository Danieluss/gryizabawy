import unittest
from unittest import mock
import numpy as np

from quests.genetic_toolbox import GeneticToolbox
from quests.individual import Individual


class GeneticToolboxTest(unittest.TestCase):

    def test_crossover(self) -> None:
        np.random.seed(13)
        world_model = mock.Mock()
        world_model.fix_individual = lambda x: x
        gt = GeneticToolbox(world_model)
        a = Individual([('a',), ('b',), ('c',)], [('c',), ('d',)])
        b = Individual([('x',), ('y',)], [('z',), ('y',)])
        c, d = gt.crossover(a, b)
        self.assertEqual(sorted(c.initial_state + d.initial_state),
                         sorted(a.initial_state + b.initial_state))
        self.assertEqual(sorted(c.final_state + d.final_state),
                         sorted(a.final_state + b.final_state))
        print(c.initial_state, c.final_state)
        print(d.initial_state, d.final_state)

    def test_mutation(self) -> None:
        np.random.seed(16)
        world_model = mock.Mock()
        world_model.fix_individual = lambda x: x
        world_model._sample_predicate = lambda: ('a',)
        gt = GeneticToolbox(world_model)
        a = Individual([('a',), ('b',), ('c',)], [('c',), ('d',)])

        b = gt.mutate(a)[0]

        self.assert_(a.initial_state == b.initial_state or a.final_state == b.final_state)
        self.assert_(abs(len(a.initial_state) - len(b.initial_state)) <= 1)
        self.assert_(abs(len(a.final_state) - len(b.final_state)) <= 1)

        print(a.initial_state, a.final_state)
        print(b.initial_state, b.final_state)
