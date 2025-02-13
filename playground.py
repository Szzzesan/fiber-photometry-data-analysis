import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Simulated nonlinear signal
t = np.linspace(0, 2, 1000)  # Time
signal = np.sin(2 * np.pi * t) + 0.2 * np.random.randn(1000)  # Example signal with noise

# Detect peaks (positive)
peaks, _ = find_peaks(signal, height=0)  # Find peaks above 0
# Detect troughs (negative)
troughs, _ = find_peaks(-signal, height=0)  # Invert signal to find troughs

# Peak-to-Trough transition
if len(peaks) > 0 and len(troughs) > 0:
    peak_idx = peaks[0]  # First peak
    trough_idx = troughs[0]  # First trough after event
    delta_P = signal[peak_idx] - signal[trough_idx]

# Plot signal
plt.figure(figsize=(10, 5))
plt.plot(t, signal, label="Signal", color='gray')
plt.scatter(t[peaks], signal[peaks], color='red', label="Peak", marker='^')
plt.scatter(t[troughs], signal[troughs], color='blue', label="Trough", marker='v')

# Draw arrow from peak to trough
if len(peaks) > 0 and len(troughs) > 0:
    plt.annotate("", xy=(t[trough_idx], signal[trough_idx]),
                 xytext=(t[peak_idx], signal[peak_idx]),
                 arrowprops=dict(arrowstyle="->", color='black'))

plt.legend()
plt.title("Peak-to-Trough Analysis")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
print('hello')