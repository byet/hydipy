import unittest
from hydipy.ParameterLearning import CPTLearner
import pandas as pd


class Test_DiscreteNode(unittest.TestCase):
    def test_counts_parents(self):
        learner = CPTLearner()
        train_data = pd.DataFrame(
            {"a": ["a1", "a2", "a1"], "b": ["b1", "b1", "b2"], "c": ["c2", "c2", "c1"]},
        )
        state_names = {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]}
        variable = "c"
        parents = ["b", "a"]
        cpt_no_correction = learner.cpt_counts(
            variable=variable,
            train_data=train_data,
            state_names=state_names,
            parents=parents,
            correction=0,
        )

        col_multi_index = pd.MultiIndex.from_tuples(
            tuples=[("b1", "a1"), ("b1", "a2"), ("b2", "a1"), ("b2", "a2")],
            names=["b", "a"],
        )
        row_index = pd.Index(["c1", "c2"], name="c")
        check_df_no_correction = pd.DataFrame(
            [[0.00, 0.00, 1.00, 0.00], [1.00, 1.00, 0.00, 0.00]],
            index=row_index,
            columns=col_multi_index,
        )

        pd.testing.assert_frame_equal(cpt_no_correction, check_df_no_correction)

        cpt_correction = learner.cpt_counts(
            variable=variable,
            train_data=train_data,
            state_names=state_names,
            parents=parents,
            correction=0.01,
        )
        check_df_correction = pd.DataFrame(
            [[0.01, 0.01, 1.01, 0.01], [1.01, 1.01, 0.01, 0.01]],
            index=row_index,
            columns=col_multi_index,
        )
        pd.testing.assert_frame_equal(cpt_correction, check_df_correction)

    def test_counts_no_parents(self):
        learner = CPTLearner()
        train_data = pd.DataFrame(
            {"a": ["a1", "a2", "a1"], "b": ["b1", "b1", "b2"], "c": ["c2", "c2", "c1"]},
        )
        state_names = {"c": ["c1", "c2"]}
        variable = "c"
        parents = []
        cpt_no_correction = learner.cpt_counts(
            variable=variable,
            train_data=train_data,
            state_names=state_names,
            parents=parents,
            correction=0,
        )

        row_index = ["c1", "c2"]
        col_index = ["c"]
        check_df_no_correction = pd.DataFrame(
            [[1.0], [2.0]], index=row_index, columns=col_index
        )

        pd.testing.assert_frame_equal(cpt_no_correction, check_df_no_correction)

        cpt_correction = learner.cpt_counts(
            variable=variable,
            train_data=train_data,
            state_names=state_names,
            parents=parents,
            correction=0.02,
        )
        check_df_correction = pd.DataFrame(
            [[1.02], [2.02]], index=row_index, columns=col_index
        )

        pd.testing.assert_frame_equal(cpt_correction, check_df_correction)
