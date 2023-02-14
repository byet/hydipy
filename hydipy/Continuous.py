import numpy as np
from pgmpy.factors.discrete import TabularCPD
from scipy.stats import uniform


class DiscretizedNode:
    """
    Base class for continuous nodes
    """
    def __init__(self, id, parents = [], disc = None):
        """_summary_

        Args:
            id (string): node id
            disc (list, optional): initial discretization points e.g. [-1, 0, 1]. Defaults to None.
        """
        if parents is not None:
            if isinstance(parents, str):
                raise TypeError(
                    "Evidence must be list, tuple or array of strings.")
        self.id = id
        self.parents = parents
        if disc is not None:
            self.set_discretization(disc)

    def set_discretization(self, disc):
        """Sets discretization points, lower bounds, upper bounds and medians of intervals

        Args:
            disc (array): disretization points e.g. array([-1, 0, 1])
        """
        self.disc = disc
        self.lb = disc[:-1]
        self.ub = disc[1:]
        self.median = (self.ub + self.lb) / 2
        state_names = np.column_stack([self.lb, self.ub])
        self.states = [",".join(item) for item in state_names.astype(str)]
        self.cardinality = len(disc) - 1

    def discretized_probability_mass(self, dist, lb, ub):
        """Builds discretization probability mass probability table based on the distribution cdf
        
        Args:
            dist (Distribution): A distribution object
            lb (array): 1d numpy array of lower bounds of discretization intervals
            ub (array): 1d numpy array of upper bounds of discretization intervals

        Returns:
            (array, array): two dimensional array of discretization intervals, and a one dimensional array of probability mass corresponding to discretiation intervals
        """
        potentials = dist.cdf(ub) - dist.cdf(lb)
        probs = potentials / potentials.sum()
        return probs

    def compute_error(self, probs, lb, ub):
        """Computes the entropy error for a discretization

        Args:
            probs (array): one dimensional array of probability mass corresponding to object intervals
            lb (array): 1d numpy array of lower bounds of intervals
            ub (array): 1d numpy array of upper bounds of intervals

        Returns:
            array: entropy error corresponding to each interval

        """
        f_lb = np.concatenate([[0], probs[:-1]])
        f_ub = np.concatenate([probs[1:], [0]])
        f_bounds = np.array([f_lb, probs, f_ub])
        f_bar = np.mean(f_bounds, axis=0)
        f_min = np.min(f_bounds, axis=0)
        f_max = np.max(f_bounds, axis=0)
        wj = ub - lb

        f_min[f_min <= 0] = 0.000000001
        f_max[f_max <= 0] = 0.000000001

        a = (f_max-f_bar)/(f_max-f_min)
        b = (f_bar-f_min)/(f_max-f_min)
        c = np.log(f_min / f_bar)
        d = np.log(f_max / f_bar)
        res = (a*f_min*c + b*f_max*d)*wj

        return res

    def set_evidence(self, value, evidence_tolerance):
        """Adds evidence interval on the current discretization and returns the addd state

        Args:
            value (float): evidence value
            evidence_tolerance (float): % evidence tolerance for the discretized evidence state, should be between 0.0 and 1.0

        Raises:
            ValueError: evidence_tolerance should be between 0.0 and 1.0

        Returns:
            string: returns the added evidence state as a string
        """
        if (evidence_tolerance >= 1) or (evidence_tolerance <= 0):
            raise ValueError("evidence tolerance should be between 0.0 and 1.0")

        evidence_bound = abs(value * evidence_tolerance)
        evidence_lb = value - evidence_bound
        evidence_ub = value + evidence_bound
        # You may as well delete all other intervals
        ev_disc = np.concatenate([self.disc[self.disc < evidence_lb], [
                                 evidence_lb, evidence_ub], self.disc[self.disc > evidence_ub]])

        self.set_discretization(ev_disc)
        return f"{evidence_lb},{evidence_ub}"

    def add_interval(self, disc, index):
        """
        Divide interval with index
        """
        added_value = (disc[index] + disc[index + 1]) / 2
        new_disc = np.insert(disc, index + 1, added_value)
        return new_disc

    def merge_interval(self, disc, index):
        """
        Remove interval with index
        """
        new_disc = np.delete(disc, index)
        return new_disc

    def update_intervals(self, probs):
        """
        Divides interval with highest error
        Merges all intervals with 0 error
        Returns the indices of added and merged intervals

        """
        current_disc = self.disc
        entropy_error = self.compute_error(probs, self.lb, self.ub)
        added_index = entropy_error.argmax()
        new_disc = self.add_interval(current_disc, added_index)

        removed_indices = np.where(entropy_error < 0.000001)[0]
        # Since added index modifies index values in entropy error, we add 1 index value

        if len(removed_indices) > 0:
            removed_indices[removed_indices >= added_index] += 1
            new_disc = self.merge_interval(new_disc, removed_indices)

        self.set_discretization(new_disc)

        return new_disc, added_index, removed_indices

    def cdf(self, x, disc=None, probs=None):
        if disc is None:
            disc = self.disc
        if probs is None:
            probs = self.probs

        new_probs = np.insert(self.probs, 0, 0.)
        cum_probs = np.cumsum(new_probs)
        return np.interp(x, xp=self.disc, fp=cum_probs)

    def ppf(self, x, disc=None, probs=None):
        if disc is None:
            disc = self.disc
        if probs is None:
            probs = self.probs
        
        new_probs = np.insert(self.probs, 0, 0.)
        cum_probs = np.cumsum(new_probs)
        return np.interp(x, xp=cum_probs, fp=self.disc)

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
        variance = (probs * (midpoints - mean)**2).sum()
        stdev = np.sqrt(variance)
        lq = self.percentile(disc, probs, lci)
        uq = self.percentile(disc, probs, uci)
        return {'mean':mean, 'std':stdev, 'lci':lq, 'hci':uq}

    def percentile(self, disc, probs, q):
        """Computes percentile using numpy interpolation function

        Args:
            disc (array): 1d array of discretization points
            probs (array): 1d array of probability mass corresponding to state intervals
            q (_type_): percentile point

        Returns:
            float: percentile point
        """
        ext_probs = np.insert(probs,0,0.)
        cum_probs = np.cumsum(ext_probs)
        return np.interp(q, xp=cum_probs, fp=disc)

    def build_pgmpy_cpd(self, parent_nodes = None):
        """builds a pgmpy cpd based on current discretization

        Args:
            parent_nodes (list, optional): list of parent node objects. Defaults to None.

        Returns:
            TabularCPD: pgmpy TabularCPD object
        """
        parents_card = []
        pgmpy_states = {}
        pgmpy_states[self.id] = self.states
        
        if parent_nodes is not None:
            for parent in parent_nodes:
                parents_card.append(parent.cardinality)
                pgmpy_states[parent.id] = parent.states
        cpd = TabularCPD(variable=self.id, variable_card=self.cardinality, values= self.values, evidence=self.parents, evidence_card=parents_card, state_names=pgmpy_states)
        
        return cpd

    def build_cpt(self, parent_nodes=[]):
        pass

    def initialize_intervals(self, parent_nodes=[]):
        pass

    def initialize_cpt(self, parent_nodes=[]):
        self.initialize_intervals(parent_nodes=parent_nodes)
        self.build_cpt(parent_nodes=parent_nodes)

    def update_cpt(self, parent_nodes=[]):
        self.build_cpt(parent_nodes=parent_nodes)


class ContinuousNode(DiscretizedNode):
    def __init__(self, id, dist=None, parents=[], disc=None):
        super().__init__(id=id, parents=parents, disc=disc)
        self.dist = dist
       
    def initialize_intervals(self, parent_nodes=[], num_init_disc_states=3, lci=0.001, uci=0.999):
        """Initalized discretization for cpt. Estimates lower and upper bound for 
        discretization and divides into equal spaces.

        Args:
            parent_nodes (list, optional): _description_. Defaults to [].
            num_init_disc_states (int, optional): number of initial intervals. Defaults to 3.
            lci (float, optional): percentile point for lower bound of discretization. Defaults to 0.001.
            uci (float, optional): percentile point for upper bound of discretization. Defaults to 0.999.

        Returns:
            array: 1d array of discretization points
        """
        if parent_nodes:
            minval = np.inf
            maxval = -np.inf
            cpt_states = self._parent_state_combinations(parent_nodes)
            for state in cpt_states:
                state_num = [[float(i) for i in x.split(",")] for x in state]
                minval = min(minval, np.min(self.dist.ppf(lci, *state_num)))
                maxval = max(maxval, np.max(self.dist.ppf(uci, *state_num)))
        else:
            minval = self.dist.ppf(lci)
            maxval = self.dist.ppf(uci)

        disc = np.linspace(minval, maxval, (num_init_disc_states + 1))
        self.set_discretization(disc)
        return disc
        
    def _parent_state_combinations(self, parent_nodes):
        parent_states = {}
        num_states = 1
        for parent in parent_nodes:
            parent_states[parent.id] = parent.states
            num_states *= parent.cardinality
        state_combinations = np.array(np.meshgrid(*parent_states.values())).T.reshape(num_states, -1)
        return state_combinations

    def build_cpt_column(self, *args):
        """Builds the conditional probability mass for one column of continuous 
        distribution with parents. Each interval of the parent need to be passed.

        Returns:
            array: 1d array of probabilities for a column of cpt
        """
        combinations = np.array(np.meshgrid(*args)).T.reshape(-1, len(args))
        potentials = np.array([self.dist.cdf(
            self.ub, *pars) - self.dist.cdf(self.lb, *pars) for pars in combinations]).sum(axis=0)
        probs = potentials / potentials.sum()
        return probs

    def build_cpt(self, parent_nodes=[]):
        """Builds cpt for continuous nodes

        Args:
            parent_nodes (list): list of parent nodes (continuous node objects). Defaults to [].

        Returns:
            array: 2d array of cpt values 
        """
        probs = []
        if not parent_nodes:
            probs = self.discretized_probability_mass(self.dist, self.lb, self.ub)
            values = np.row_stack(probs)
        else:
            cpt_states = self._parent_state_combinations(parent_nodes)
            for cpt_state in cpt_states:
                state_num = [[float(i) for i in x.split(",")] for x in cpt_state]
                prob = self.build_cpt_column(*state_num)
                probs.append(prob)
            values = np.column_stack(probs)
        self.values = values
        return values

class Deterministic(ContinuousNode):
    def __init__(self, id, expression=None, parents=[], disc=None):
        super().__init__(id=id, parents=parents, disc=disc)
        self.expression = expression
      
    def initialize_intervals(self, parent_nodes=[], num_init_disc_states=3):
        """Initalized discretization for cpt. Estimates lower and upper bound for 
        discretization and divides into equal spaces.

        Args:
            parent_nodes (list, optional): _description_. Defaults to [].
            num_init_disc_states (int, optional): number of initial intervals. Defaults to 3.
            lci (float, optional): percentile point for lower bound of discretization. Defaults to 0.001.
            uci (float, optional): percentile point for upper bound of discretization. Defaults to 0.999.

        Returns:
            array: 1d array of discretization points
        """
        if not parent_nodes:
            raise ValueError("Deterministic nodes must have parents.")
        minval = np.inf
        maxval = -np.inf
        cpt_states = self._parent_state_combinations(parent_nodes)
        for state in cpt_states:
            state_num = [[float(i) for i in x.split(",")] for x in state]
            combinations = np.array(np.meshgrid(*state_num)).T.reshape(-1, len(state_num))
            minval = min(minval, np.min(
                [self.expression(*comb) for comb in combinations]))
            maxval = max(minval, np.max(
                [self.expression(*comb) for comb in combinations]))

        disc = np.linspace(minval, maxval, (num_init_disc_states + 1))
        self.set_discretization(disc)
        return disc

    def build_cpt_column(self, *args):
        """Builds the conditional probability mass for one column of continuous 
        distribution with parents. Each interval of the parent need to be passed.

        Returns:
            array: 1d array of probabilities for a column of cpt
        """
        combinations = np.array(np.meshgrid(*args)).T.reshape(-1, len(args))
        results = np.array([self.expression(*pars) for pars in combinations])

        # scipy uniform parameters are lower bound and interval width
        loc = np.min(results)
        scale = np.max(results) - loc
        uniform_pars = [loc, scale]
        potentials = uniform.cdf(self.ub, *uniform_pars) - uniform.cdf(self.lb, *uniform_pars)
        probs = potentials / potentials.sum()
        return probs

class MixtureNode(DiscretizedNode):
    def __init__(self, id, values=None, parents=[], disc=None):
        super().__init__(id=id, parents=parents, disc=disc)
        dists = np.array(values)
        if dists.ndim != 2:
            raise TypeError("Values must be a 2D list/array")
        self.dists = dists

    
    def initialize_intervals(self, parent_nodes=[], num_init_disc_states=3, lci=0.001, uci=0.999):
        """Initalized discretization for cpt. Estimates lower and upper bound for 
        discretization and divides into equal spaces.

        Args:
            parent_nodes (list, optional): List of continuous parent nodes Defaults to [].
            num_init_disc_states (int, optional): number of initial intervals. Defaults to 3.
            lci (float, optional): percentile point for lower bound of discretization. Defaults to 0.001.
            uci (float, optional): percentile point for upper bound of discretization. Defaults to 0.999.

        Returns:
            array: 1d array of discretization points
        """
        minval = np.inf
        maxval = -np.inf
        for dist in self.dists:
            minval = min(minval, dist[0].ppf(lci))
            maxval = max(maxval, dist[0].ppf(uci))
        disc = np.linspace(minval, maxval, (num_init_disc_states + 1))
        self.set_discretization(disc)
        return disc

    def build_cpt_column(self, dist):
        return self.discretized_probability_mass(dist, self.lb, self.ub)

    def build_cpt(self, parent_nodes=[]):
        """Builds cpt for mixture nodes

        Args:
            parent_nodes (list): list of continuosu parent nodes. Defaults to [].

        Returns:
            array: 2d array of cpt values 
        """
        probs = []
        for dist in self.dists:
            prob = self.build_cpt_column(dist[0])
            probs.append(prob)
        values = np.column_stack(probs)
        self.values = values
        return values
