import unittest
import numpy as np

from quests.pddl_solver import PDDLSolver
from quests.xml_parser import XmlParser
from quests.world_model import WorldModel


class WorldModelTest(unittest.TestCase):

    def test_transtion(self) -> None:
        np.random.seed(12)
        wm = WorldModel(XmlParser("data/geneticquest_db.xml"), PDDLSolver(), 30, 10)
        ind = wm.sample_individual()
        print(ind.initial_state)
        print("****************")
        print(ind.final_state)
        print(wm.transition_to_state(ind))
