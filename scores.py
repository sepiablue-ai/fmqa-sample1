import numpy as np

def compute_Q(remaining, utilization, centroid_dist,
              M=63, d_max=1.0,
              w_r=100.0, w_u=10.0, w_c=5.0):
    """
    remaining: int >=0
    utilization: 0..1 (float)
    centroid_dist: Euclidean distance from bin center (same units as d_max)
    Returns Q >= 0 (smaller is better)
    """
    r_norm = remaining / float(M)            # 0..1
    u = 1.0 - float(utilization)             # 0..1 (smaller is better)
    d_norm = min(1.0, centroid_dist / float(d_max))  # clip to [0,1]
    Q = w_r * r_norm + w_u * u + w_c * d_norm
    return float(Q)

def compute_transformed_score(pack_result, remaining, utilization, centroid_dist,
                              M=63, d_max=1.0,
                              w_r=100.0, w_u=10.0, w_c=5.0,
                              alpha_exp=1.0, cm_override=None, Q_init_mean=None):
    """
    Returns transformed target y_bar = -exp(-Q / c_m)
    If cm_override is given, use it; else need Q_init_mean (mean Q over initial dataset) to set c_m.
    """
    Q = compute_Q(remaining, utilization, centroid_dist, M, d_max, w_r, w_u, w_c)
    if cm_override is not None:
        c_m = cm_override
    else:
        if Q_init_mean is None:
            # fallback
            c_m = 1.0
        else:
            c_m = alpha_exp * float(Q_init_mean)
            if c_m <= 0:
                c_m = 1.0
    y_bar = -np.exp(- Q / c_m)
    return y_bar, Q