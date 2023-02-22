import pandas as pd 

class CPTLearner:
    def __init__(self, train_data=None):
        self.train_data = train_data

    def cpt_counts(self, variable, train_data, state_names, parents = [], ignore_na=True, correction = 0.01):
        """_summary_

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
            data = train_data.dropna()
        else:
            data = train_data

        if not parents:
            counts = data.loc[:, variable].value_counts().astype(float)
            state_counts = (
                counts.reindex(state_names[variable])
                .fillna(0)
                .to_frame()
            )

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
            state_counts = counts.reindex(index=row_index, columns=column_index).fillna(0) + correction
        return state_counts
        
