import adi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Sweep Parameters
START_FREQ = 300e6
STOP_FREQ = 800e6
STEP_FREQ = 5e6
SWEEP_DELAY = 0.1
FFT_SIZE = 1024
TRAIL_LENGTH = 10  # Number of previous sweeps to display

# SDR Parameters
SAMPLE_RATE = 2e6
BANDWIDTH = 20e6
GAIN = 30

# Frequency sweep list
frequencies = np.arange(START_FREQ, STOP_FREQ + STEP_FREQ, STEP_FREQ)
freq_index = [0]

# Initialize SDR
sdr = adi.Pluto("ip:192.168.2.1")
sdr.rx_sample_rate = int(SAMPLE_RATE)
sdr.rx_rf_bandwidth = int(BANDWIDTH)
sdr.rx_gain_control_mode = "manual"
sdr.rx_hardwaregain = GAIN
sdr.rx_lo = int(frequencies[0])

# Relative and absolute frequency axes
rel_freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / SAMPLE_RATE))

# Setup trailing buffer
spectrum_trail = []  # List of (freq_axis, spectrum) tuples

# Setup plot
fig, ax = plt.subplots()
ax.set_title("Sweeping Spectrum Monitor (Pluto SDR)")
ax.set_xlabel("Absolute Frequency (MHz)")
ax.set_ylabel("Power (dB)")
ax.set_ylim(-100, 0)
ax.set_xlim(START_FREQ / 1e6, STOP_FREQ / 1e6)
ax.grid(True)

# Prepare initial line objects for the trail
lines = [ax.plot([], [], color='blue', alpha=0.1)[0] for _ in range(TRAIL_LENGTH)]

def update(frame):
    current_freq = frequencies[freq_index[0] % len(frequencies)]
    sdr.rx_lo = int(current_freq)
    freq_index[0] += 1

    samples = sdr.rx()
    if len(samples) < FFT_SIZE:
        return lines

    windowed = samples[:FFT_SIZE] * np.hanning(FFT_SIZE)
    spectrum = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(windowed))) + 1e-10)

    abs_freq_axis = (rel_freq_axis + current_freq) / 1e6

    # Update trail buffer
    spectrum_trail.append((abs_freq_axis, spectrum))
    if len(spectrum_trail) > TRAIL_LENGTH:
        spectrum_trail.pop(0)

    # Update plot lines
    for i, (freqs, spec) in enumerate(spectrum_trail):
        lines[i].set_data(freqs, spec)
        lines[i].set_alpha((i + 1) / TRAIL_LENGTH)  # Increasing opacity

    ax.set_title(f"Spectrum Sweep - LO @ {current_freq / 1e6:.1f} MHz")
    return lines

ani = animation.FuncAnimation(fig, update, interval=SWEEP_DELAY * 1000, blit=True)
plt.tight_layout()
plt.show()

# Cleanup on exit
sdr.rx_destroy_buffer()
