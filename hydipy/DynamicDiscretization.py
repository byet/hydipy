from hydipy.Continuous import MixtureNode, ContinuousNode
from hydipy.Discrete import DiscreteNode
import warnings
import pyAgrum as gum
import matplotlib.pyplot as plt
import numpy as np
from networkx import DiGraph, topological_sort


class Hybrid_BN:
    def __init__(self, cpts):
        self.edges = []
        self.nodes = {}
        self.nodes = self._init_nodes(cpts)
        self.edges = self._init_edges(cpts)
        # Bn to get partial order
        self.partial_order = self.topological_order(self.edges)
        self.partial_order_cont_nodes = self.sort_continuous_nodes(
            self.partial_order, self.nodes
        )
        self.aux_bn_cpds = self._init_cpts(self.nodes, self.partial_order)
        self.aux_bn = self._create_pyagrum_bn(self.nodes, self.aux_bn_cpds)

        self.propagated = False

    def _init_nodes(self, cpts):
        nodes = {}
        for cpt in cpts:
            variable = cpt.id
            nodes[variable] = cpt
        return nodes

    def _init_edges(self, cpts):
        edges = []
        for cpt in cpts:
            edges += cpt.agrum_edges()
        return edges

    def _create_pyagrum_bn(self, nodes, agrum_cpds={}):
        bn = gum.BayesNet()
        for node in nodes.values():
            bn.add(node.agrum_var())
        bn.addArcs(self.edges)
        for node_name in nodes.keys():
            cpd_probs = agrum_cpds[node_name]
            bn.cpt(node_name).fillWith(cpd_probs)
        return bn

    def topological_order(self, edges):
        return list(topological_sort(DiGraph(edges)))

    def _init_cpts(self, nodes, partial_order):
        cpds = {}
        for variable in partial_order:
            node = nodes[variable]
            if not isinstance(node, (DiscreteNode, MixtureNode, ContinuousNode)):
                raise ValueError(
                    "Only DiscreteNode, MixtureNode or ContinuousNode can be added."
                )

            parent_nodes = [nodes[parent] for parent in node.parents]

            if isinstance(node, (MixtureNode, ContinuousNode)):
                node.initialize_cpt(parent_nodes)
            cpd_probs = node.agrum_cpd()
            cpds[variable] = cpd_probs
        return cpds

    def update_cont_cpt(self, variable, nodes):
        node = nodes[variable]
        if not isinstance(node, (MixtureNode, ContinuousNode)):
            raise ValueError(
                "Only discretizations of MixtureNode or ContinuousNode can be updated."
            )
        parent_nodes = [nodes[parent] for parent in node.parents]
        node.update_cpt(parent_nodes)
        cpd_probs = node.agrum_cpd()
        self.aux_bn_cpds[variable] = cpd_probs
        return cpd_probs

    def reset_aux_bn(self):
        cpds = self._init_cpts(self.nodes, self.partial_order)
        self.aux_bn_cpds = cpds
        self.aux_bn = self._create_pyagrum_bn(self.nodes, cpds)
        self.propagated = False

    def refresh_aux_bn(self):
        self.aux_bn = self._create_pyagrum_bn(self.nodes, self.aux_bn_cpds)
        self.propagated = False

    def sort_continuous_nodes(self, partial_order, nodes):
        partial_order_cont_nodes = []
        for variable in partial_order:
            if isinstance(nodes[variable], (ContinuousNode, MixtureNode)):
                partial_order_cont_nodes.append(variable)
        return partial_order_cont_nodes


class DynamicDiscretization:
    def __init__(self, hbn, evidence_tolerance=0.01):
        self.model = hbn
        self.evidence_tolerance = evidence_tolerance

    def query(
        self, variables, evidence=None, n_iter=10, show_stats=False, show_figures=False
    ):
        aux_evidence = None
        if evidence is not None:
            aux_evidence = evidence.copy()
        continuous_nodes_ordered = self.model.partial_order_cont_nodes

        if self.model.propagated:
            self.model.reset_aux_bn()

        if evidence is not None and set(continuous_nodes_ordered).intersection(
            evidence.keys()
        ):
            for variable in continuous_nodes_ordered:
                if variable in evidence.keys():
                    evidence_value = evidence[variable]
                    node = self.model.nodes[variable]
                    aux_ev_state = node.set_evidence(
                        float(evidence_value), self.evidence_tolerance
                    )
                    aux_evidence[variable] = aux_ev_state
            # Update all cpts after entering evidence
            for variable in continuous_nodes_ordered:
                node = self.model.nodes[variable]
                self.model.update_cont_cpt(variable, self.model.nodes)
            self.model.refresh_aux_bn()

        # Select which nodes to apply DD
        if not continuous_nodes_ordered:
            warnings.warn("No Continuous or Mixture nodes available in this model.")
        elif evidence:
            queried_cont_nodes = [
                node for node in continuous_nodes_ordered if node not in evidence.keys()
            ]
        else:
            queried_cont_nodes = continuous_nodes_ordered

        # DD
        for iter in range(n_iter):

            infer = gum.LazyPropagation(self.model.aux_bn)
            infer.setEvidence(aux_evidence)
            infer.makeInference()
            # Update intervals of non-evidence nodes
            for variable in queried_cont_nodes:
                node = self.model.nodes[variable]
                marginal = infer.posterior(variable).toarray()
                node.update_intervals(marginal)
            # Update cpts of all continuous nodes (as evidence nodes parents may change)
            for variable in continuous_nodes_ordered:
                node = self.model.nodes[variable]
                self.model.update_cont_cpt(variable, self.model.nodes)
            self.model.refresh_aux_bn()

        infer = gum.LazyPropagation(self.model.aux_bn)
        infer.setEvidence(aux_evidence)
        infer.makeInference()
        self.model.propagated = True

        if show_stats:
            self.query_variable_stats(variables, infer, show_figures)
        return infer

    def query_variable_stats(self, variables, posteriors, show_figures=False):

        for variable in variables:
            post = posteriors.posterior(variable)
            post_values = post.toarray()
            node = self.model.nodes[variable]
            print(f"***** Variable: {variable} ******\n")
            if isinstance(node, DiscreteNode):
                print(post)
                if show_figures:
                    states = node.states
                    plt.bar(states, post_values)
            else:
                lci = 0.05
                uci = 0.95
                stats = self.summary_stats(node.disc, post_values, lci, uci)
                print(f"Mean: {stats['mean']:.3f}, Std. Dev: {stats['std']:.3f}")
                print(
                    f"Percentile ({lci * 100}% — {uci * 100}%) = ({stats['lci']:.3f} — {stats['uci']:.3f})"
                )
                if show_figures:
                    fig, ax = plt.subplots()
                    bins = node.disc
                    probs = post_values
                    ax.hist(bins[:-1], bins, weights=probs)
                    ax.set_title(variable)

    def summary_stats(self, disc, probs, lci=0.05, uci=0.95):
        """Summary statistics of a discretized distribution

        Args:
            disc (array): 1d array of discretization points
            probs (array): 1d array of probability mass corresponding to state intervals
            lci (float, optional): lower credible interval percentile. Defaults to 0.05.
            uci (float, optional): upper credible interval percentile. Defaults to 0.95.

        Returns:
            dict: mean, standard deviation, lower and upper credible interval points
        """
        midpoints = (disc[:-1] + disc[1:]) / 2
        mean = (midpoints * probs).sum()
        variance = (probs * (midpoints - mean) ** 2).sum()
        stdev = np.sqrt(variance)
        lq = self.percentile(disc, probs, lci)
        uq = self.percentile(disc, probs, uci)
        return {"mean": mean, "std": stdev, "lci": lq, "uci": uq}

    def percentile(self, disc, probs, q):
        """Computes percentile using numpy interpolation function

        Args:
            disc (array): 1d array of discretization points
            probs (array): 1d array of probability mass corresponding to state intervals
            q (_type_): percentile point

        Returns:
            float: percentile point
        """
        ext_probs = np.insert(probs, 0, 0.0)
        cum_probs = np.cumsum(ext_probs)
        return np.interp(q, xp=cum_probs, fp=disc)
