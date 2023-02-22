import pandas as pd 
from hydipy.Discrete import DiscreteNode

class CPTLearner:
    def __init__(self, model, train_data=None, correction=0.01):
        self.model = model
        self.train_data = train_data
        self.correction = correction

    def cpt_counts(self, variable, state_names, parents = [], ignore_na=True):
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

        if ignore_na:
            data = self.train_data.dropna()
        else:
            data = self.train_data

        if not parents:
            counts = data.loc[:, variable].value_counts().astype(float)
            state_counts = (
                counts.reindex(state_names[variable])
                .fillna(0)
                .to_frame()
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
            state_counts = counts.reindex(index=row_index, columns=column_index).fillna(0) + self.correction
        return state_counts

    def learn_cpt(self, node, normalize=True):
        node_id = node.id
        parents = node.parents
        state_names = dict()
        state_names[node_id] = node.states
        if parents:
            state_names.update({parent:self.model.nodes[parent].states for parent in parents})
        counts = self.cpt_counts(variable=node_id, state_names=state_names, parents=parents).to_numpy()
        node = DiscreteNode(id=node_id, values=counts, parents=parents, states=state_names[node_id])
        if normalize:
            node.normalize()
        return node
    
    def mle(self):
        return {key:self.learn_cpt(node) for (key, node) in self.model.nodes.items()}


        
