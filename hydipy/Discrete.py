from pgmpy.factors.discrete import TabularCPD

class Discrete:
    def __init__(self, id, values, parents, states):
        self.id = id
        self.state_names = states
        self.parents = parents
        self.values = values
        self.cardinality = len(values)
        self.states = states
    def build_pgmpy_cpd(self, parents = None):
        parents_card = []
        pgmpy_states = self.states.copy()
        if parents is not None:
            for parent in parents:
                parents_card.append(parent.cardinality)
                pgmpy_states.update(parent.states)
        cpd = TabularCPD(variable=self.id, variable_card=self.cardinality, values= self.values, evidence=self.parents, evidence_card=parents_card, state_names=pgmpy_states)
        self.cpd = cpd
        return cpd

