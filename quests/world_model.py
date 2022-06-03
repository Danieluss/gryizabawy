from collections import defaultdict
from typing import List, Tuple

import numpy as np

from quests.individual import Individual
from quests.pddl_solver import PDDLSolver
from quests.xml_parser import XmlParser


class WorldModel:
    """Class containing plot creation logic.
    It maintains current global state and creates quests candidates (individuals).
    It is responsible for keeping the states consistent.
    """
    def __init__(self, parser: XmlParser, solver: PDDLSolver,
                 initial_state_n: int, final_state_n: int,
                 best_pattern: List[int] = [1, 1, 1, -1]) -> None:
        self.initial_state_n = initial_state_n
        self.final_state_n = final_state_n
        self.parser = parser
        self.solver = solver
        self.objects, self.objects_by_type, self.type_by_object \
            = parser.get_objects()
        self.initial_state = parser.get_initial_state()
        self.predicates, self.predicates_dict = \
            parser.get_predicates()
        self.actions = parser.get_actions()
        self.actions_tension = parser.get_tension()
        self.best_pattern = np.cumsum(best_pattern)

    def _sample_predicate(self) -> Tuple[str]:
        """Samples a single predicates from a list of available predicates
        and fills it with objects of appropriate type.
        """

        predicate = np.random.choice(self.predicates)
        objects = []
        for parameter in predicate['parameters']:
            objects.append(np.random.choice(self.objects_by_type[parameter['type']]))
        return (predicate['name'], *objects)

    def _sample_state(self, n_max: int) -> List[Tuple[str]]:
        """Samples state"""

        n = np.random.randint(0, n_max)+1
        return [self._sample_predicate() for _ in range(n)]

    def sample_individual(self) -> Individual:
        """Samples individual"""

        return self.fix_individual(Individual(
            self._sample_state(self.initial_state_n),
            self._sample_state(self.final_state_n)
        ))

    def _fix_state(self, state: List[Tuple[str]]) -> List[Tuple[str]]:
        """Fixes state. Removes duplicate and opposite predicates.
        Removes excess predicates when an argument has to be unique.
        Ordering is used to resolve conflicts. When there is a contradiction
        between predicates, predicate that is earlier on the list is kept.
        """

        shortened_state = []
        existing_predicates = defaultdict(lambda: set())
        for predicate in state:
            name = predicate[0]
            key = name + "_" + "_".join([self.type_by_object[p] for p in predicate[1:]])
            if key not in self.predicates_dict:
                shortened_state.append(predicate)
                continue
            rules = self.predicates_dict[key]
            if predicate[1:] in existing_predicates[name]:
                continue
            if rules['opposite'] is not None and \
                    predicate[1:] in existing_predicates[rules['opposite']]:
                continue
            flag = False
            for i, parameter in enumerate(rules['parameters']):
                if parameter.get('unique') == 'true' and \
                        f"{predicate[i+1]}@{i}" in existing_predicates[name]:
                    flag = True
                    break
            if flag:
                continue
            existing_predicates[name].add(predicate[1:])
            shortened_state.append(predicate)
            for i, parameter in enumerate(rules['parameters']):
                if parameter.get('unique') == 'true':
                    existing_predicates[name].add(f"{predicate[i+1]}@{i}")

        return shortened_state

    def fix_individual(self, individual: Individual) -> Individual:
        """Fixes both states of an individual.
        """

        return Individual(
            self._fix_state(individual.initial_state),
            self._fix_state(individual.final_state)
        )

    def _run_solver(self, individual: Individual) -> List[Tuple[str, ...]]:
        """Finds a path to the final state of an individual starting from
        the union of the global initial state and individual's initial state.
        """

        return self.solver.solve(
            self._fix_state(self.initial_state+individual.initial_state),
            individual.final_state,
            self.objects_by_type,
            self.actions,
        )

    def _evaulate_actions(self, actions_taken: List[Tuple[str, ...]]) -> np.ndarray:
        """Computes tension of each action in the plot."""

        current_pattern = []
        for action in actions_taken:
            current_pattern.append(self.actions_tension[action[0]])
        return np.array(current_pattern)

    def evaluate_individual(self, individual: Individual) -> float:
        """Computes fitness of an individual.
        It is defined as plot_length/(mse(plot_tension, desired_tension)+0.1).
        Plot is generated using pddl solver.
        """

        # TODO think about extensions if the plot is long
        actions_taken = self._run_solver(individual)
        if len(actions_taken) == 0:
            return 0.0
        # print(actions_taken)
        current_pattern = self._evaulate_actions(actions_taken)
        current_pattern = np.cumsum(current_pattern)
        n = np.lcm(len(self.best_pattern), len(current_pattern))
        current_pattern = np.repeat(current_pattern, n/len(current_pattern))
        best_pattern = np.repeat(self.best_pattern, n/len(self.best_pattern))
        # print(current_pattern, best_pattern)
        mse = np.mean((current_pattern - best_pattern)**2)
        # print(mse)
        return len(actions_taken)/(mse+0.1)

    def transition_to_state(self, individual: Individual) \
            -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
        """Returns a new global state that is obtained after
        applying actions that lead to individual final state.
        """

        actions_taken = self._run_solver(individual)
        current_state = set(self._fix_state(self.initial_state+individual.initial_state))
        for action in actions_taken:
            action_definition = self.actions[action[0]]
            args_mapping = {}
            for p, arg in zip(action_definition["parameters"], action[1:]):
                args_mapping[p[0]] = arg
            for effect in action_definition["effects"]:
                not_flag = False
                if effect[0] == "not":
                    effect = effect[1:]
                    not_flag = True
                predicate = [effect[0]]
                for p in effect[1:]:
                    predicate.append(args_mapping[p])
                if not_flag:
                    current_state.remove((*predicate,))
                else:
                    current_state.add((*predicate,))

        return list(current_state), actions_taken

    def update_initial_state(self, initial_state: List[Tuple[str, ...]]) -> None:
        """Updates world model initial state."""
        self.initial_state = initial_state
