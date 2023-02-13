import numbers
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from scipy.stats import uniform


class DiscretizedNode:
    """
    Base class for continuous nodes
    """

    def set_discretization(self, disc):
        """Sets discretization points, lower bounds, upper bounds and medians of intervals

        Args:
            disc (array): disretization points e.g. array([-1, 0, 1])
        """
        self.disc = disc
        self.lb = disc[:-1]
        self.ub = disc[1:]
        self.median = (self.ub + self.lb) / 2

    def build_prob(self, dist):
        """Builds discretization intervals and probability mass probability table based on the distribution cdf
        Args:
            dist (Distribution): A distribution object

        Returns:
            (array, array): two dimensional array of discretization intervals, and a one dimensional array of probability mass corresponding to discretiation intervals
        """
        potentials = dist.cdf(self.ub) - dist.cdf(self.lb)
        probs = potentials / potentials.sum()
        intervals = np.column_stack([self.lb, self.ub])
        return intervals, probs

    def compute_error(self, probs):
        """Computes the entropy error for a discretization

        Args:
            probs (array): one dimensional array of probability mass corresponding to object intervals

        Returns:
            array: entropy error corresponding to each interval

        """
        f_lb = np.concatenate([[0], probs[:-1]])
        f_ub = np.concatenate([probs[1:], [0]])
        f_bounds = np.array([f_lb, probs, f_ub])
        f_bar = np.mean(f_bounds, axis=0)
        f_min = np.min(f_bounds, axis=0)
        f_max = np.max(f_bounds, axis=0)
        wj = self.ub - self.lb

        f_min[f_min <= 0] = 0.000000001
        f_max[f_max <= 0] = 0.000000001

        a = (f_max-f_bar)/(f_max-f_min)
        b = (f_bar-f_min)/(f_max-f_min)
        c = np.log(f_min / f_bar)
        d = np.log(f_max / f_bar)
        res = (a*f_min*c + b*f_max*d)*wj

        return res

    def set_evidence(self, value, evidence_threshold):
        """
        Set evidence interval on the current discretization
        Returns evidence state
        """
        evidence_lb = value - evidence_threshold
        evidence_ub = value + evidence_threshold
        # You may as well delete all other intervals
        ev_disc = np.concatenate([self.disc[self.disc < evidence_lb], [
                                 evidence_lb, evidence_ub], self.disc[self.disc > evidence_ub]])

        self.set_discretization(ev_disc)
        return f"{evidence_lb},{evidence_ub}"

    def add_interval(self, index):
        """
        Divide interval with index
        """
        added_value = (self.disc[index] + self.disc[index + 1]) / 2
        new_disc = np.insert(self.disc, index + 1, added_value)
        self.set_discretization(new_disc)
        return new_disc

    def merge_interval(self, index):
        """
        Remove interval with index
        """
        new_disc = np.delete(self.disc, index)
        self.set_discretization(new_disc)
        return (new_disc)

    def update_intervals(self, probs):
        """
        Divides interval with highest error
        Merges all intervals with 0 error
        Returns the indices of added and merged intervals

        """
        self.probs = probs
        entropy_error = self.compute_error(probs)
        self.entropy_error = entropy_error
        added_index = entropy_error.argmax()
        new_disc = self.add_interval(added_index)

        removed_indices = np.where(entropy_error < 0.000001)[0]

        # Since added index modifies index values in entropy error, we add 1 index value

        if len(removed_indices) > 0:
            removed_indices[removed_indices >= added_index] += 1
            new_disc = self.merge_interval(removed_indices)

        return new_disc, added_index, removed_indices

    def cdf(self, x):
        new_probs = np.insert(self.probs, 0, 0.)
        cum_probs = np.cumsum(new_probs)
        return np.interp(x, xp=self.disc, fp=cum_probs)

    def ppf(self, x):
        new_probs = np.insert(self.probs, 0, 0.)
        cum_probs = np.cumsum(new_probs)
        return np.interp(x, xp=cum_probs, fp=self.disc)

    def summary_stats(self, intervals, probs, lci=0.05, uci=0.95):
        midpoints = intervals.mean(axis=1)
        mean = (midpoints * probs).sum()
        variance = (probs * (midpoints - mean)**2).sum()
        stdev = np.sqrt(variance)
        lq = self.quantile(intervals, probs, lci)
        uq = self.quantile(intervals, probs, uci)
        return mean, stdev, lq, uq

    def quantile(self, intervals, probs, q):
        cumdist = np.cumsum(probs)
        index = np.argwhere(cumdist >= q)[0]
        lprob = 0 if index == 0 else cumdist[index - 1]
        uprob = cumdist[index]
        interpolvalue = intervals[index][0][0] + (q - lprob) * (
            intervals[index][0][1] - intervals[index][0][0]) / (uprob - lprob)
        return interpolvalue

    def build_tabular_cpd(self):
        state_names = dict()
        state_names[self.variable] = [
            ",".join(item) for item in self.state_names.astype(str)]
        if self.par_state_names:
            state_names.update(self.par_state_names)
        return TabularCPD(variable=self.variable, variable_card=len(self.disc) - 1,
                          values=self.values,
                          evidence=self.evidence,
                          evidence_card=self.evidence_card,
                          state_names=state_names)


class MixtureNode(DiscretizedNode):
    def __init__(self, variable, values, evidence=None, evidence_card=None, par_state_names=None):
        self.variable = variable
        self.par_state_names = par_state_names
        self.evidence = evidence
        self.evidence_card = evidence_card

        if evidence_card is not None:
            if isinstance(evidence_card, numbers.Real):
                raise TypeError("Evidence card must be a list of numbers")

        if evidence is not None:
            if isinstance(evidence, str):
                raise TypeError(
                    "Evidence must be list, tuple or array of strings.")
            if not len(evidence_card) == len(evidence):
                raise ValueError(
                    "Length of evidence_card doesn't match length of evidence"
                )

        dists = np.array(values)
        if dists.ndim != 2:
            raise TypeError("Values must be a 2D list/array")

        self.dists = dists
        self.initialize_discretization()
        self.build_cpt()

    def initialize_discretization(self):
        minval = np.inf
        maxval = -np.inf
        for dist in self.dists:
            minval = min(minval, dist[0].ppf(0.005))
            maxval = max(maxval, dist[0].ppf(0.995))
        disc = np.linspace(minval, maxval, 4)
        self.disc = disc
        return disc

    def build_cpt(self):
        values = []
        npar = len(self.dists)
        nchi = len(self.disc) - 1
        self.set_discretization(self.disc)
        for dist in self.dists:
            _, probs = self.build_prob(dist[0])
            values = np.append(values, probs)
        self.state_names = np.column_stack([self.lb, self.ub])
        self.values = values.reshape(npar, nchi).T


class ContinuousNode(DiscretizedNode):
    def __init__(self, variable, dist, evidence=None):

        if evidence is not None:
            if isinstance(evidence, str):
                raise TypeError(
                    "Evidence must be list, tuple or array of strings.")

        self.variable = variable
        self.evidence = evidence

        self.dist = dist
        self.disc = np.linspace(-2, 2, 4)
        self.probs = np.ones(4-1) / (4-1)

        self.set_discretization(self.disc)
        self.state_names = np.column_stack([self.lb, self.ub])
        # self.build_cpt()

    def initialize_discretization(self, parent_states=None):
        if parent_states:
            minval = np.inf
            maxval = -np.inf
            states = list(parent_states.values())

            num_states = 1
            for el in states:
                num_states *= len(el)

            for index, row in enumerate(states):
                states[index] = [element for element in row]
            cpt_states = np.array(np.meshgrid(
                *states)).T.reshape(num_states, -1)
            for state in cpt_states:
                state = [[float(i) for i in x.split(",")] for x in state]
                minval = min(minval, np.min(self.dist.ppf(0.005, *state)))
                maxval = max(maxval, np.max(self.dist.ppf(0.995, *state)))

        else:
            minval = self.dist.ppf(0.001)
            maxval = self.dist.ppf(0.999)

        disc = np.linspace(minval, maxval, 4)
        self.disc = disc
        self.set_discretization(self.disc)
        return disc

    # TODO what if no parents

    def build_cpt_interval(self, *args):
        combinations = np.array(np.meshgrid(*args)).T.reshape(-1, len(args))
        potentials = np.array([self.dist.cdf(
            self.ub, *pars) - self.dist.cdf(self.lb, *pars) for pars in combinations]).sum(axis=0)
        probs = potentials / potentials.sum()
        intervals = np.column_stack([self.lb, self.ub])
        return intervals, probs

    def build_cpt(self, par_states=None):
        evidence_card = []
        if not par_states:
            intervals, probs = self.build_prob(self.dist)
            values = np.row_stack(probs)
        else:
            # par_states = {'x': ['0.4,0.5', '0.5,0.6'], 'y':['0.05,0.1','0.1,0.2']}
            states = list(par_states.values())

            num_states = 1
            for el in states:
                num_states *= len(el)

            for index, row in enumerate(states):
                states[index] = [element for element in row]

            probs = []

            cpt_states = np.array(np.meshgrid(
                *states)).T.reshape(num_states, -1)
            for cpt_state in cpt_states:

                state = [[float(i) for i in x.split(",")] for x in cpt_state]
                num_par = len(state)
                _, prob = self.build_cpt_interval(*state)
                probs.append(prob)

            values = np.column_stack(probs)
            for parent in self.evidence:
                evidence_card.append(len(par_states[parent]))

        self.par_state_names = par_states
        self.state_names = np.column_stack([self.lb, self.ub])
        self.values = values
        self.evidence_card = evidence_card


class Deterministic(ContinuousNode):
    def __init__(self, variable, expression, evidence=None):
        if evidence is not None:
            if isinstance(evidence, str):
                raise TypeError(
                    "Evidence must be list, tuple or array of strings.")

        self.variable = variable
        self.evidence = evidence

        self.expression = expression
        self.disc = np.linspace(-2, 2, 4)
        self.probs = np.ones(4-1) / (4-1)

        self.set_discretization(self.disc)
        self.state_names = np.column_stack([self.lb, self.ub])
        # self.build_cpt()

    def initialize_discretization(self, parent_states=None):
        if not parent_states:
            raise ValueError("Deterministic nodes must have parents")

        minval = np.inf
        maxval = -np.inf
        states = list(parent_states.values())

        num_states = 1
        for el in states:
            num_states *= len(el)

        for index, row in enumerate(states):
            states[index] = [element for element in row]
        cpt_states = np.array(np.meshgrid(
            *states)).T.reshape(num_states, -1)

        for state in cpt_states:
            state = [[float(i) for i in x.split(",")] for x in state]
            combinations = np.array(np.meshgrid(
                *state)).T.reshape(-1, len(state))
            minval = min(minval, np.min(
                [self.expression(*comb) for comb in combinations]))
            maxval = max(minval, np.max(
                [self.expression(*comb) for comb in combinations]))

        disc = np.linspace(minval, maxval, 4)
        self.disc = disc
        self.set_discretization(self.disc)
        return disc

    def build_cpt_interval(self, *args):
        combinations = np.array(np.meshgrid(*args)).T.reshape(-1, len(args))
        results = np.array([self.expression(*pars) for pars in combinations])
        # scipy uniform parameters are lower bound and interval width
        loc = np.min(results)
        scale = np.max(results) - loc
        uniform_pars = np.array([loc, scale])
        potentials = uniform.cdf(self.ub, *uniform_pars) - \
            uniform.cdf(self.lb, *uniform_pars)
        probs = potentials / potentials.sum()
        intervals = np.column_stack([self.lb, self.ub])
        return intervals, probs

    def build_cpt(self, par_states=None):
        evidence_card = []
        if not par_states:
            raise ValueError("Deterministic nodes must have parents")

        # par_states = {'x': ['0.4,0.5', '0.5,0.6'], 'y':['0.05,0.1','0.1,0.2']}
        states = list(par_states.values())

        num_states = 1
        for el in states:
            num_states *= len(el)

        for index, row in enumerate(states):
            states[index] = [element for element in row]

        probs = []

        cpt_states = np.array(np.meshgrid(
            *states)).T.reshape(num_states, -1)

        for cpt_state in cpt_states:

            state = [[float(i) for i in x.split(",")] for x in cpt_state]
            num_par = len(state)
            _, prob = self.build_cpt_interval(*state)
            probs.append(prob)

        values = np.column_stack(probs)
        for parent in self.evidence:
            evidence_card.append(len(par_states[parent]))

        self.par_state_names = par_states
        self.state_names = np.column_stack([self.lb, self.ub])
        self.values = values
        self.evidence_card = evidence_card
