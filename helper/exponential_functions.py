import numpy as np

def exp_decreasing(x, cumulative=8., starting=1.):
    a = starting
    b = a / cumulative
    density = a / np.exp(b * x)
    return density

def get_equal_area_bins(num_bins: int, cumulative: float = 8.0, starting: float = 1.0) -> np.ndarray:
    """
    Calculates x-bin edges for an exponential function so each bin has equal area.
    """
    b = starting / cumulative
    probabilities = np.linspace(0, 1, num_bins + 1)

    # The inverse CDF is x(p) = -ln(1-p) / b
    # handle p=1 separately since log(0) is -infinity
    # The x-value for p=1 is theoretically infinity
    with np.errstate(divide='ignore'): # Ignore divide by zero warning for log(0)
        bin_edges = -np.log(1 - probabilities) / b
    bin_edges[0] = 0
    return bin_edges