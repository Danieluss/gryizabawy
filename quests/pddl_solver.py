import os
import subprocess as sp
from typing import Dict, List, Tuple
import uuid

from quests.utils.file_utils import save_file


class PDDLSolver:
    """PDDLSolver with its main method solve(...) creates PDDL domain and problem
    and runs hsp2 planner to solve it. The solve(...) method return the generated plan.
    """
    def __init__(self, time_limit: int = 5000, path: str = "tmp") -> None:
        self.time_limit = time_limit
        self.path = path

    def _generate_actions(self, actions: Dict[str, Dict]) -> Tuple[str, Dict[str, int]]:
        actions_pddl = []
        predicates_num_args = {}
        for name, action in actions.items():
            action_pddl = f"(:action {name}\n"

            parameters = []
            for parameter in action['parameters']:
                parameters.append(f"?{parameter[0]} - {parameter[1]}")

            preconditions = []
            for precondition in action['preconditions']:
                precondition_parameters = []
                for parameter in precondition[1:]:
                    precondition_parameters.append("?" + parameter)
                preconditions.append(f"({precondition[0]} {' '.join(precondition_parameters)})")
                predicates_num_args[precondition[0]] = len(precondition_parameters)

            effects = []
            for effect in action['effects']:
                start = 1
                if effect[0] == "not":
                    start += 1
                effect_parameters = []
                for parameter in effect[start:]:
                    effect_parameters.append("?" + parameter)
                effect_str = f"({effect[start-1]} {' '.join(effect_parameters)})"
                if effect[0] == "not":
                    effect_str = f"(not {effect_str})"
                effects.append(effect_str)
                predicates_num_args[effect[start-1]] = len(effect_parameters)

            action_pddl += f"\t:parameters ({' '.join(parameters)})\n"
            action_pddl += f"\t:precondition (and {' '.join(preconditions)})\n"
            action_pddl += f"\t:effect (and {' '.join(effects)})\n"
            action_pddl += ")\n"
            actions_pddl.append(action_pddl)

        return '\n'.join(actions_pddl), predicates_num_args

    def _generate_domain(self, objects_by_type: Dict[str, List[str]],
                         actions: Dict[str, Dict]) -> str:
        domain = f"(define (domain world)\n"
        domain += "  (:requirements :typing)\n"
        domain += f"  (:types {' '.join(objects_by_type.keys())})\n"
        actions_pddl, predicates_num_args = self._generate_actions(actions)
        domain += f"  (:predicates\n"
        for predicate, n in predicates_num_args.items():
            domain += f"    ({predicate} " + " ".join("?" + chr(97+i) for i in range(n)) + ")\n"
        domain += f"  )\n"
        domain += actions_pddl
        domain += "\n)"

        return domain

    def _generate_problem(self, initial_state: List[Tuple[str, ...]],
                          final_state: List[Tuple[str, ...]],
                          objects_by_type: Dict[str, List[str]]) -> str:
        problem = f"(define (problem quests)\n"
        problem += f"  (:domain world)\n"
        problem += f"  (:objects\n"
        for type, objects in objects_by_type.items():
            problem += f"    {' '.join(objects)} - {type}\n"
        problem += "  )\n"

        problem += "  (:init\n"
        problem += "\n    ".join("(" + " ".join(predicate) + ")" for predicate in initial_state)
        problem += "  )\n"

        problem += "  (:goal (and\n"
        problem += "\n    ".join("(" + " ".join(predicate) + ")" for predicate in final_state)
        problem += "  ))\n"

        problem += ")"

        return problem

    def solve(self, initial_state: List[Tuple[str, ...]],
              final_state: List[Tuple[str, ...]],
              objects_by_type: Dict[str, List[str]],
              actions:  Dict[str, Dict]) -> List[Tuple[str, ...]]:
        path = self.path + "/" + str(uuid.uuid4())

        save_file(self._generate_domain(objects_by_type, actions), path + "_domain.pddl")

        save_file(self._generate_problem(initial_state, final_state, objects_by_type),
                  path + "_problem.pddl")

        args = [
            "./hsp-planners/hsp2-1.0/bin/hsp2",
            "-S",
            f"[backward,h1plus,{self.time_limit}]",
            f"{path}_problem.pddl",
            f"{path}_domain.pddl"
        ]
        with sp.Popen(args, stdout=sp.PIPE, stderr=sp.DEVNULL) as process:
            solver_result = process.stdout.read()

        actions_taken = []
        for line in solver_result.decode("utf-8").split("\n"):
            if not line.startswith("QUESTS"):
                continue
            for action in line.split(","):
                if action[0] == "(" and action[-1] == ")":
                    actions_taken.append(action[1:-1].lower().split(" "))

        os.system(f"rm {path}_problem.pddl {path}_domain.pddl")

        return actions_taken
