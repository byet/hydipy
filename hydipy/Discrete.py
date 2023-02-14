from pgmpy.factors.discrete import TabularCPD

class DiscreteNode:
    def __init__(self, id, values, parents, states):
        self.id = id
        self.state_names = states
        self.parents = parents
        self.values = values
        self.cardinality = len(values)
        self.states = states
    def build_pgmpy_cpd(self, parent_nodes = None):
        parents_card = []
        pgmpy_states = {}
        pgmpy_states[self.id] = self.states
        if parent_nodes is not None:
            for parent in parent_nodes:
                parents_card.append(parent.cardinality)
                pgmpy_states[parent.id] = parent.states
        cpd = TabularCPD(variable=self.id, variable_card=self.cardinality, values= self.values, evidence=self.parents, evidence_card=parents_card, state_names=pgmpy_states)
        self.cpd = cpd
        return cpd

