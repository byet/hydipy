import pandas as pd
from hydipy.Discrete import DiscreteNode
from hydipy.DynamicDiscretization import Hybrid_BN
import pyAgrum as gum
from math import log


class CPTLearner:
    def __init__(self, model, train_data=None, correction=0.01):
        self.model = model
        self.train_data = train_data
        self.correction = correction

    def cpt_counts(self, variable, state_names, parents=[], ignore_na=True):
        """Counts the values for a CPT in the dataset

        Args:
            variable (string): variable name
            train_data (DataFrame): pandas dataframe of train_data. Missing values are marked as na.
            state_names (dict): A dictionary of state_names. e.g.  {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]}
            parents (list, optional): list of parent names . Defaults to [].
            ignore_na (bool, optional): drops na values from data if True. Defaults to True.

        Returns:
            _type_: _description_
        """

        cpt_vars = [variable] + parents

        if ignore_na:
            data = self.train_data[cpt_vars].dropna()
        else:
            data = self.train_data[cpt_vars]

        if not parents:
            counts = data.loc[:, variable].value_counts().astype(float)
            state_counts = (
                counts.reindex(state_names[variable]).fillna(0).to_frame()
            ) + self.correction

        else:
            parents_states = [state_names[parent] for parent in parents]
            counts = data.groupby([variable] + parents).size().unstack(parents)

            if not isinstance(counts.columns, pd.MultiIndex):
                counts.columns = pd.MultiIndex.from_arrays([counts.columns])

            # reindex rows & columns to sort them and to add missing ones
            # missing row    = some state of 'variable' did not occur in data
            # missing column = some state configuration of current 'variable's parents
            #                  did not occur in data
            row_index = state_names[variable]
            column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = (
                counts.reindex(index=row_index, columns=column_index).fillna(0)
                + self.correction
            )
        return state_counts

    def _get_node_info(self, node):
        node_id = node.id
        parents = node.parents
        state_names = dict()
        state_names[node_id] = node.states
        if parents:
            state_names.update(
                {parent: self.model.nodes[parent].states for parent in parents}
            )
        return node_id, state_names, parents

    def _get_non_missing_counts(self, node, normalize=True):
        node_id, state_names, parents = self._get_node_info(node)
        return self.cpt_counts(
            variable=node_id, state_names=state_names, parents=parents
        ).to_numpy()

    def learn_cpt(self, node, normalize=True):
        counts = self._get_non_missing_counts(node)
        node_id, state_names, parents = self._get_node_info(node)
        new_node = DiscreteNode(
            id=node_id, values=counts, parents=parents, states=state_names[node_id]
        )
        if normalize:
            new_node.normalize()
        return new_node

    def mle(self):
        return {key: self.learn_cpt(node) for (key, node) in self.model.nodes.items()}


class EMLearner(CPTLearner):
    def __init__(self, model, train_data=None, correction=0.01):
        self.model = model
        self.train_data = train_data
        self.correction = correction
        self.missing_data = self.train_data[self.train_data.isnull().any(axis=1)]
        self.nrows_missing = self.missing_data.shape[0]
        self.missing_var_sets = self.missing_variables()
        self.ie = None

    def missing_variables(self):
        missing_vars = self.train_data.columns[self.train_data.isna().any()]
        missing_var_sets = dict()
        for var in missing_vars:
            nodes_in_cpt = [var] + self.model.nodes[var].parents
            missing_var_sets[var] = nodes_in_cpt
        return missing_var_sets

    def prepare_inference_engine(self):
        ie = gum.LazyPropagation(self.model.aux_bn)
        for key, nodes in self.missing_var_sets.items():
            ie.addJointTarget(set(nodes))
        return ie

    def init_full_counts(self):
        return {
            key: self._get_non_missing_counts(node)
            for (key, node) in self.model.nodes.items()
        }

    def get_posterior(self, ie, vars_in_cpt):
        post = ie.jointPosterior(set(vars_in_cpt)).reorganize(vars_in_cpt).toarray()
        return post.T.reshape(2, -1)

    def define_cpt(self, node, counts, normalize=True):
        node_id, state_names, parents = self._get_node_info(node)
        new_node = DiscreteNode(
            id=node_id, values=counts, parents=parents, states=state_names[node_id]
        )
        if normalize:
            new_node.normalize()
        return new_node

    def e_step(self, counts):
        ie = self.prepare_inference_engine()
        log_prob = 0
        expected_counts = {key: value.copy() for key, value in counts.items()}
        for i in range(self.nrows_missing):
            row = self.missing_data.iloc[i]
            missing_vars_in_row = self.missing_data.columns[row.isna()]
            # We need expected counts for missing variables and children of missing variables
            vars_needing_expected_counts = [
                key
                for key, value in self.missing_var_sets.items()
                if set(value).intersection(set(missing_vars_in_row))
            ]
            evidence = row.dropna().to_dict()
            ie.setEvidence(evidence)
            log_prob += log(ie.evidenceProbability())
            for variable in vars_needing_expected_counts:
                vars_in_cpt = self.missing_var_sets[variable]
                post = self.get_posterior(ie, vars_in_cpt)
                expected_counts[variable] += post
        return expected_counts, log_prob

    def m_step(self, counts):
        return {
            key: self.define_cpt(node, counts[key])
            for (key, node) in self.model.nodes.items()
        }

    def log_likelihood_data(self, data):
        ie = self.prepare_inference_engine()
        log_prob = 0
        nrows = data.shape[0]
        for i in range(nrows):
            row = data.iloc[i]
            evidence = row.dropna().to_dict()
            ie.setEvidence(evidence)
            log_prob += log(ie.evidenceProbability())
        return log_prob

    def em(self, num_iter=20, log_lik_threshold=1e-5):
        full_counts = self.init_full_counts()
        m_cpts = dict()
        log_prob = 0
        for i in range(num_iter):
            print(
                f"Iteration: {i+1} {'' if log_prob == 0 else f'; Log-likelihood: {log_prob:.5f}'}",
                end="\r",
            )
            new_log_prob = self.log_likelihood_data(self.train_data.dropna())
            e_counts, missing_log_prob = self.e_step(full_counts)
            new_log_prob += missing_log_prob
            m_cpts = self.m_step(e_counts)
            # Stopping rule
            if abs(new_log_prob - log_prob) < log_lik_threshold:
                return m_cpts
            model = Hybrid_BN(m_cpts.values())
            self.model = model
            log_prob = new_log_prob
        return m_cpts
