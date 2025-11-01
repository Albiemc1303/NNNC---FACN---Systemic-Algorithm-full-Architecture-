# systemic_symmetry_algorithm.py
SYMMETRY_EIGENVALUE_THRESHOLD = 0.12
CHC_SCORE_THRESHOLD = 0.75

def symmetry_integrity_check(laplacian_eigenvalues):
    if len(laplacian_eigenvalues) < 2:
        return False
    lambda_2 = sorted(laplacian_eigenvalues)[1]
    return float(lambda_2) >= SYMMETRY_EIGENVALUE_THRESHOLD

def chc_trigger(chc_score):
    return float(chc_score) >= CHC_SCORE_THRESHOLD