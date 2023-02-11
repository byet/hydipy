import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from networkx import topological_sort
from hydipy.ContinuousNodes import MixtureNode, ContinuousNode


class HybridBayesianNetwork(BayesianNetwork):
    def __init__(self, ebunch=None, latents=set()):
        super(HybridBayesianNetwork, self).__init__(
            ebunch=ebunch, latents=latents)
        self.contnodes = []
        self.disc_par_cont_nodes = []
        self.cont_par_cont_nodes = []

    def add_cpds(self, *cpds):
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, MixtureNode, ContinuousNode)):
                raise ValueError(
                    "Only TabularCPD or PartitionedExpression can be added.")
            elif isinstance(cpd, MixtureNode):
                self.disc_par_cont_nodes.append(cpd)
                self.contnodes.append(cpd.variable)
            elif isinstance(cpd, ContinuousNode):
                self.cont_par_cont_nodes.append(cpd)
                self.contnodes.append(cpd.variable)
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node):
        allcpds = self.cpds + self.disc_par_cont_nodes + self.cont_par_cont_nodes
        if node is not None:
            for cpd in allcpds:
                if cpd.variable == node:
                    return cpd
        else:
            return allcpds

    def topological_order(self):
        return list(topological_sort(self))


class DD:
    def __init__(self, hbn):
        self.model = hbn
        self.evidence_threshold = 0.01

    def get_parent_states(self, cpd, aux_bn):
        parents = cpd.evidence
        par_states = {}
        for parent in parents:
            par_states.update(aux_bn.get_cpds(parent).state_names)
        return par_states

    def query(self, variables, evidence=None, n_iter=10):

        aux_evidence = evidence
        self.aux_bn = BayesianNetwork(self.model.edges)
        self.aux_bn.add_cpds(*self.model.cpds)
        top_order = self.model.topological_order()
        top_order_cont = [
            node for node in top_order if node in self.model.contnodes]

        # Initialize Discretization
        for node in top_order_cont:
            cpd = self.model.get_cpds(node)
            if isinstance(cpd, ContinuousNode) and cpd.evidence:
                par_states = self.get_parent_states(cpd, self.aux_bn)
                cpd.initialize_discretization(par_states)

                if node in evidence:
                    evidence_value = evidence[node]
                    aux_ev_state = cpd.set_evidence(
                        float(evidence_value), self.evidence_threshold)
                    aux_evidence[node] = aux_ev_state
                cpd.build_cpt(par_states)

            else:
                cpd.initialize_discretization()
                if node in evidence:
                    evidence_value = evidence[node]
                    aux_ev_state = cpd.set_evidence(
                        float(evidence_value), self.evidence_threshold)
                    aux_evidence[node] = aux_ev_state
                cpd.build_cpt()

            aux_cpd = cpd.build_tabular_cpd()
            self.aux_bn.add_cpds(aux_cpd)

        # Select which nodes to apply DD
        if not self.model.contnodes:
            print("No cont nodes")
        elif evidence:
            query_cont_nodes = [
                node for node in top_order_cont if node not in evidence.keys()]
        else:
            query_cont_nodes = top_order_cont

        # DD
        for iter in range(n_iter):
            infer = VariableElimination(self.aux_bn)
            cont_marginals = infer.query(
                query_cont_nodes, evidence=aux_evidence, joint=False)

            for node in query_cont_nodes:
                cpd = self.model.get_cpds(node)
                marginal = cont_marginals[node].values
                cpd.update_intervals(marginal)

            for node in top_order_cont:
                cpd = self.model.get_cpds(node)
                if isinstance(cpd, ContinuousNode) and cpd.evidence:
                    par_states = self.get_parent_states(cpd, self.aux_bn)
                    cpd.build_cpt(par_states)
                else:
                    cpd.build_cpt()
                aux_cpd = cpd.build_tabular_cpd()
                self.aux_bn.add_cpds(aux_cpd)

        infer = VariableElimination(self.aux_bn)
        # print(variables)
        # print(aux_evidence)
        return infer.query(variables, evidence=aux_evidence, joint=False)


def summarize(states, probs, lci=0.05, uci=0.95):
    intervals = np.array([[float(i) for i in x.split(",")] for x in states])
    return summary_stats(intervals, probs, lci, uci)


def summary_stats(intervals, probs, lci=0.05, uci=0.95):
    midpoints = intervals.mean(axis=1)
    mean = (midpoints * probs).sum()
    variance = (probs * (midpoints - mean)**2).sum()
    stdev = np.sqrt(variance)
    lq = float(quantile(intervals, probs, lci))
    uq = float(quantile(intervals, probs, uci))
    return mean, stdev, (lq, uq)


def quantile(intervals, probs, q):
    cumdist = np.cumsum(probs)
    index = np.argwhere(cumdist >= q)[0]
    lprob = 0 if index == 0 else cumdist[index - 1]
    uprob = cumdist[index]
    interpolvalue = intervals[index][0][0] + (q - lprob) * (
        intervals[index][0][1] - intervals[index][0][0]) / (uprob - lprob)
    return interpolvalue
