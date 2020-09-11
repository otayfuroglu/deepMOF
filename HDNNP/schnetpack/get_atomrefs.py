#
import numpy as np

n_elements = 118
atomizationEs = {
    1: -13.568416326526803,
    6: -1027.290084166281,
    8: -2038.792640354206,
    30: -48405.07779371168
}

idx_atomizationE = np.zeros((n_elements, 1))
for idx, E in atomizationEs.items():
    idx_atomizationE[idx] = E

def atomrefs_energy0(energy_keyword):
    return {energy_keyword: idx_atomizationE}

