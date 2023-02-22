from pyAgrum import LabelizedVariable
import numpy as np


class DiscreteNode:
    def __init__(self, id, values, parents, states):
        self.id = id
        self.states = states
        self.parents = parents
        self.values = np.array(values)
        self.cardinality = len(values)

    def normalize(self):
        """Normalized CPDs so each column sum to 1

        Returns:
            array: normalized cpd
        """
        normalized = self.values / self.values.sum(axis=0)

        # Replace nan with uniform probabilities
        uniform_prob = 1 / normalized.shape[0]
        normalized = np.nan_to_num(normalized, nan=uniform_prob)

        self.values = normalized
        return normalized

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

    def __eq__(self, other):
        return (
            self.id == other.id
            and set(self.states) == set(other.states)
            and set(self.parents) == set(other.parents)
            and np.allclose(self.values, other.values)
        )
