# dydipy - Hybrid BNs with Dynamic Discretization

Supports mixture nodes and continuous nodes with continuous parents.

~~~python
from DD import HybridBayesianNetwork, DD
from pgmpy.factors.discrete import TabularCPD
from ContinuousNodes import MixtureNode
from Distributions import NormalDistribution
from networkx import topological_sort



model = HybridBayesianNetwork([('x1','y'),('x2','y')])
x1_cpd = TabularCPD(variable='x1', variable_card=2,
                   values=[[0.6],
                           [0.4]], state_names={'x1':['a','b']})
x2_cpd = TabularCPD(variable='x2', variable_card=2,
                   values=[[0.1],
                           [0.9]], state_names={'x2':['d','e']})


y_cpd = MixtureNode(variable='y', values=[[NormalDistribution(5, 1)],[NormalDistribution(1, 5)],[NormalDistribution(3, 2)],[NormalDistribution(0, 4)]], evidence=['x1','x2'], evidence_card=[2,2], par_state_names= {'x1':['a','b'], 'x2':['d','e']})
print(list(topological_sort(model)))
print(model.topological_order())

#print(y_cpd.values)
#print(y_cpd.state_names)

model.add_cpds(x1_cpd, x2_cpd, y_cpd)
print(model.get_cpds('y'))


dd = DD(model)


print(dd.model.get_cpds('y').values)
print(dd.model.get_cpds('y').state_names)

print(dd.query(['x2'], evidence={'y':'-1'})['x2'])
print(dd.query(['y'], evidence={})['y'])
~~~