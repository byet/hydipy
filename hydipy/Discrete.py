from pyAgrum import LabelizedVariable
import numpy as np


class DiscreteNode:
    def __init__(self, id, values, parents, states):
        self.id = id
        self.states = states
        self.parents = parents
        self.values = np.array(values)
        self.cardinality = len(values)

    def agrum_var(self):
        return LabelizedVariable(self.id, self.id, self.states)

    def agrum_edges(self):
        edges = []
        if self.parents:
            # reversed because of variable ordering differences in pyagrum
            for parent in reversed(self.parents):
                edges.append((parent, self.id))
        return edges

    def agrum_cpd(self):
        probs = self.values.T.reshape(-1)
        return probs
