from pgmpy.models import BayesianNetwork
from itertools import product
import pandas as pd
from pgmpy.inference import VariableElimination
import numpy as np
import networkx as nx

def format_monotonic_influences(C, D, names):
  return [
    f"{names[j]} ≺ᴹ⁺ {names[i]} : {-1 * D[i, j]:0.2f}"
    if C[i, j] == +1
    else f"{names[j]} ≺ᴹ⁻ {names[i]} : {-1 * D[i, j]:0.2f}"
    for i, j in zip(*np.nonzero(C))
  ]



def compute_monotonic_influences_from_bn(model: BayesianNetwork, frame: pd.DataFrame, r: list, sign: int, epsilon: float):
  rows = []
  names = frame.columns.tolist()
  inference = VariableElimination(model)
  all_pairs = set()
  for node in model.nodes():
    for descendant in nx.descendants(model, node):
       all_pairs.add((node, descendant))

  for second, first in all_pairs:
    if first == second: continue
    numerator = inference.query([first, second], show_progress=False)
    denominator = inference.query([second], show_progress=False)
    frange = list(range(r[names.index(first)]))
    srange = list(range(r[names.index(second)]))
    terms = np.array([
      np.cumsum([numerator.get_value(**{first: fval, second: sval}) / denominator.get_value(**{second: sval})
                 for fval in frange[:-1]])
      for sval in srange
    ]).T
    #  first is influenced by second

    diffs = np.fromiter((
      sign * (row[vj2] - row[vj1])
      for row in terms
      for vj2, vj1 in product(srange, srange)
      if vj2 > vj1
    ), dtype=float)

    C = np.all((diffs) + epsilon < 0)
    degree = C * np.sum(diffs) / len(srange)
    rows.append((first, second, degree))
  return pd.DataFrame(rows, columns=("First", "Second", "Degree"))
