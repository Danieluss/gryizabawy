import argparse
import copy
from quests.genetic_toolbox import GeneticToolbox
from quests.pddl_solver import PDDLSolver
from quests.quest_generator import QuestGenerator

from quests.utils.file_utils import read_yaml, save_json
from quests.world_model import WorldModel
from quests.xml_parser import XmlParser


def run(args) -> None:
    config = read_yaml(args.config)
    parser = XmlParser(**config["parser_args"])
    solver = PDDLSolver()
    world_model = WorldModel(parser, solver, **config["world_model_args"])
    genetic_toolbox = GeneticToolbox(world_model)
    quest_generator = QuestGenerator(**config["quest_generator_args"])

    history = []

    for _ in range(config["n_quests"]):
        best_ind, log = quest_generator.run(world_model, genetic_toolbox)
        new_state, actions_taken = world_model.transition_to_state(best_ind)
        world_model.update_initial_state(new_state)

        print(best_ind.initial_state)
        print("*******************")
        print(best_ind.final_state)
        print("*******************")
        print(actions_taken)
        print("fitness:", best_ind.fitness.values[0])

        history.append({
            "ind_initial_state": copy.deepcopy(best_ind.initial_state),
            "ind_final_state": copy.deepcopy(best_ind.final_state),
            "fitness": best_ind.fitness.values[0],
            "actions": copy.deepcopy(actions_taken),
            "final_state": copy.deepcopy(world_model.initial_state),
            "max": [x["max"] for x in log],
            "min": [x["min"] for x in log],
            "avg": [x["avg"] for x in log],
            "stddev": [x["stddev"] for x in log],
        })
        save_json(history, config["path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to a config file")
    args = parser.parse_args()
    run(args)
