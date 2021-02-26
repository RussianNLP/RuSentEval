import numpy as np


def Bootstrap(data_A, data_B, n, R):
    """
    repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
    let r be the number of times that delta(A, B) < 2*orig_delta(A, B). significance level: r/R
    this implementation follows the description in Berg-Kirkpatrick et al. (2012),
    "An Empirical Investigation of Statistical Significance in NLP".
    """
    delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)])) / n
    r = 0
    for x in range(0, R):
        temp_A = []
        temp_B = []
        samples = np.random.randint(0, n, n)
        for samp in samples:
            temp_A.append(data_A[samp])
            temp_B.append(data_B[samp])
        delta = float(sum([x - y for x, y in zip(temp_A, temp_B)])) / n
        if delta < 2 * delta_orig:
            r = r + 1
    pval = float(r) / (R)
    return pval
