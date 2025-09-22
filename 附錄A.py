import matplotlib.pyplot as plt
import numpy as np

# 設定更鮮明的顏色組合
colors_non_fatigue = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
colors_fatigue = ['red', 'magenta', 'darkorange', 'deeppink']

x = np.linspace(0, 100, 100)

def simulate_emg_vivid(base_level, peak_phase, noise=5, amplitude=20):
    y = base_level + amplitude * np.exp(-((x - peak_phase) ** 2) / 200)
    return y + np.random.normal(0, noise, size=x.shape)

emg_data_non_fatigue = {
    "Biceps brachii": simulate_emg_vivid(20, 80),
    "Triceps brachii": simulate_emg_vivid(25, 70),
    "Pectoralis major": simulate_emg_vivid(30, 75),
    "Anterior deltoid": simulate_emg_vivid(35, 85),
    "Middle deltoid": simulate_emg_vivid(25, 80),
    "Posterior deltoid": simulate_emg_vivid(20, 70),
    "Infraspinatus": simulate_emg_vivid(15, 85),
    "Latissimus dorsi": simulate_emg_vivid(18, 90),
}

emg_data_fatigue = {
    k: simulate_emg_vivid(v.mean(), peak_phase=85, noise=8, amplitude=25)
    for k, v in emg_data_non_fatigue.items()
}

groups = [
    ["Biceps brachii", "Triceps brachii"],
    ["Pectoralis major", "Anterior deltoid"],
    ["Middle deltoid", "Posterior deltoid"],
    ["Infraspinatus", "Latissimus dorsi"],
]

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
for idx, (ax, muscles) in enumerate(zip(axes, groups)):
    for i, muscle in enumerate(muscles):
        ax.plot(x, emg_data_non_fatigue[muscle], label=f"{muscle} (non-fatigue)",
                color=colors_non_fatigue[i % len(colors_non_fatigue)], linestyle='--')
        ax.plot(x, emg_data_fatigue[muscle], label=f"{muscle} (fatigue)",
                color=colors_fatigue[i % len(colors_fatigue)], linestyle='-')
    ax.set_ylabel("Activation (% MVIC)")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Cycle (%)")
fig.suptitle("Simulated EMG Comparison of Muscle Activation\nUnder Fatigue vs Non-Fatigue (Reconstructed from Zhou, 2022)", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
