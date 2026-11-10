import matplotlib.pyplot as plt
import numpy as np

# 例：精度やスコアの比較
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
experiment1 = [0.85, 0.82, 0.88, 0.85]
experiment2 = [0.90, 0.87, 0.85, 0.86]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, experiment1, width, label='実験1', color='skyblue')
bars2 = ax.bar(x + width/2, experiment2, width, label='実験2', color='lightcoral')

# 値をバーの上に表示
for i, (v1, v2) in enumerate(zip(experiment1, experiment2)):
    ax.text(i - width/2, v1 + 0.01, f'{v1:.2f}', ha='center', va='bottom')
    ax.text(i + width/2, v2 + 0.01, f'{v2:.2f}', ha='center', va='bottom')

ax.set_xlabel('評価指標')
ax.set_ylabel('スコア')
ax.set_title('実験結果の比較')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
