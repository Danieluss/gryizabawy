from collections import defaultdict
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


class XmlParser:
    """XmlParser can be used for parsing the world definition from an xml file
    Every method is responsible for extracting different element of the world,
    i.e. actions, initial state (properties and relations),
    available predicates (semantic integrity constraints),
    actions (planning operators), tension of actions.
    """

    def __init__(self, xml_path: str) -> None:
        self.world = ET.parse(xml_path).getroot()
        self.name = self.world.attrib.get('name', 'world')

    def get_objects(self) -> Tuple[List[Dict[str, str]], Dict[str, List[str]], Dict[str, str]]:
        tree = self.world.find('objects')
        objects = []
        objects_by_type = defaultdict(lambda: [])
        type_by_object = {}
        for object in tree:
            objects.append(object.attrib)
            objects_by_type[object.attrib['type']].append(object.attrib['name'])
            type_by_object[object.attrib['name']] = object.attrib['type']

        return objects, objects_by_type, type_by_object

    def get_initial_state(self) -> List[Tuple[str, ...]]:
        tree = self.world.find('relations')
        state = []
        for predicate in tree:
            values = []
            for parameter in predicate:
                values.append(parameter.attrib['value'])
            state.append((predicate.attrib['name'], *values))
        return state

    def get_predicates(self) -> List[Dict]:
        tree = self.world.find('predicates')
        predicates = []
        predicates_dict = {}
        for predicate in tree:
            name = predicate.attrib['name']
            parameters = []
            for parameter in predicate:
                name += "_" + parameter.attrib['type']
                parameters.append(parameter.attrib)

            predicates_dict[name] = {
                "parameters": parameters,
                "opposite": predicate.attrib.get('oposite')
            }
            predicates.append({
                "parameters": parameters,
                "name": predicate.attrib['name']
            })
        return predicates, predicates_dict

    def get_actions(self) -> Dict[str, Dict]:
        tree = self.world.find('operators')
        actions = {}

        for operator in tree:

            parameters = []
            for parameter in operator.find('parameters'):
                parameters.append((parameter.attrib['name'], parameter.attrib['type']))

            preconditions = []
            for precondition in operator.find('preconditions'):
                precondition_parameters = [precondition.attrib['predicate']]
                for parameter in precondition:
                    precondition_parameters.append(parameter.attrib['name'])
                preconditions.append((*precondition_parameters,))

            effects = []
            for effect in operator.find('effects'):
                effect_parameters = []
                if effect.attrib.get('negation', 'false') == 'true':
                    effect_parameters.append("not")
                effect_parameters.append(effect.attrib['predicate'])
                for parameter in effect:
                    effect_parameters.append(parameter.attrib['name'])
                effects.append((*effect_parameters,))

            actions[operator.attrib['name']] = {
                "parameters": parameters,
                "preconditions": preconditions,
                "effects": effects
            }

        return actions

    def get_tension(self) -> Dict[str, int]:
        tree = self.world.find('eventeffects')
        tension = {}
        sign_tension_mapping = {
            '+': 1,
            '=': 0,
            '-': -1
        }
        for event in tree:
            tension[event.attrib['name']] = sign_tension_mapping[event.attrib['tension']]
        return tension
