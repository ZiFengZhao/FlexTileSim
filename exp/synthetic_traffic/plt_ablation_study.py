import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150
})

# phase: [cycle_accurate, traffic_aware, static, bandwidth_distance]
latency_dict = {
    "phase1": [50.12, 51.94, 59.51, 48.75],  
    "phase2": [62.73, 68.32, 59.51, 48.75],   
    "phase3": [54.52, 53.95, 59.51, 48.75],   
    "phase4": [115.25, 120.56, 59.51, 48.75]
}

phase_labels = [
    "steady uniform, low inj",
    "steady uniform, mid inj",
    "low bursty, low inj",
    "mid bursty, mid inj"
]

model_names = ["Proposed Model", "Static Model", "Zero-Load Model"]
colors = ['#D6E3CC', '#A2D0A4', '#7BC6BE']

n_phases = len(latency_dict)
n_models = len(model_names)

ca_values = [latency_dict[f"phase{i+1}"][0] for i in range(n_phases)]
errors = [] 

for i in range(n_phases):
    ca = ca_values[i]
    phase_errors = []
    for j in range(n_models):
        pred = latency_dict[f"phase{i+1}"][j+1]
        rel_err = (pred - ca) / ca * 100  
        phase_errors.append(rel_err)
    errors.append(phase_errors)

errors = np.array(errors)

fig, ax = plt.subplots(figsize=(3.5, 2.0))  

x = np.arange(n_phases)
width = 0.18 
offsets = [-width, 0, width]  

for i, (name, color, offset) in enumerate(zip(model_names, colors, offsets)):
    bars = ax.bar(x + offset, errors[:, i], width, label=name, color=color, edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)

for i in range(n_phases):
    for j in range(n_models):
        err = errors[i, j]
        x_pos = x[i] + offsets[j]
        y_pos = err + (1.5 if err >= 0 else -4.5)  
        va = 'bottom' if err >= 0 else 'top'
        ax.text(x_pos, y_pos, f'{err:.1f}%', ha='center', va=va, fontsize=4.5, color='black')

ax.set_xticks(x)
ax.set_xticklabels(phase_labels, rotation=15, ha='right', fontsize=6)
ax.set_ylabel('Relative Error (%)', fontsize=7)
ax.set_xlabel('Traffic Phase', fontsize=7)

ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black', fontsize=6)

plt.tight_layout(pad=0.8)

plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('ablation_study.svg')
