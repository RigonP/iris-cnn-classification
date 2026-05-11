import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_iris_arrays

# Konfigurimi i stilit të grafikëve
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Ngarkimi i dataset-it IRIS nga dataset/iris.data (burimi i Moodle)
X, y, feature_names, target_names = load_iris_arrays()

# Krijimi i një DataFrame për lehtësi në manipulim
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# 1. Statistika përshkruese
print("=" * 60)
print("STATISTIKA PËRSHKRUESE TË DATASET-IT IRIS")
print("=" * 60)
print(df.groupby('species').describe().T)
print("\n")

# 2. Class distribution
plt.figure(figsize=(8, 6))
species_counts = df['species'].value_counts()
plt.bar(species_counts.index, species_counts.values, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
plt.xlabel('Iris Species', fontsize=12, fontweight='bold')
plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
plt.title('Class Distribution in the IRIS Dataset', fontsize=14, fontweight='bold')
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('iris_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Box plots për çdo veçori
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(feature_names):
    df.boxplot(column=feature, by='species', ax=axes[idx])
    axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Speciet', fontsize=10)
    axes[idx].set_ylabel('Vlera (cm)', fontsize=10)
    axes[idx].get_figure().suptitle('')

plt.suptitle('Distribution of features by species', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Pairplot - Vizualizimi çiftorë i veçorive
pairplot_fig = sns.pairplot(df, hue='species',
                            palette={'setosa': '#FF6B6B',
                                     'versicolor': '#4ECDC4',
                                     'virginica': '#95E1D3'},
                            diag_kind='kde',
                            markers=['o', 's', 'D'],
                            plot_kws={'alpha': 0.6, 's': 50})
pairplot_fig.fig.suptitle('Vizualizimi çiftorë i veçorive të IRIS dataset-it',
                          fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Heatmap i korrelacionit
plt.figure(figsize=(10, 8))
correlation_matrix = df[feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrica e korrelacionit të veçorive', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Violin plots për veçoritë kryesore (Petal Length dhe Petal Width)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.violinplot(data=df, x='species', y='petal length (cm)', ax=axes[0],
               palette=['#FF6B6B', '#4ECDC4', '#95E1D3'])
axes[0].set_title('Gjatësia e petales sipas specieve', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Speciet', fontsize=11)
axes[0].set_ylabel('Gjatësia e petales (cm)', fontsize=11)

sns.violinplot(data=df, x='species', y='petal width (cm)', ax=axes[1],
               palette=['#FF6B6B', '#4ECDC4', '#95E1D3'])
axes[1].set_title('Gjerësia e petales sipas specieve', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Speciet', fontsize=11)
axes[1].set_ylabel('Gjerësia e petales (cm)', fontsize=11)

plt.tight_layout()
plt.savefig('iris_violin_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 2D scatter plot for the most discriminative features
plt.figure(figsize=(10, 8))
colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#95E1D3'}
markers = {'setosa': 'o', 'versicolor': 's', 'virginica': 'D'}

for species in target_names:
    species_data = df[df['species'] == species]
    plt.scatter(species_data['petal length (cm)'],
                species_data['petal width (cm)'],
                c=colors[species],
                marker=markers[species],
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5,
                label=species)

plt.xlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
plt.ylabel('Petal Width (cm)', fontsize=12, fontweight='bold')
plt.title('Class Separation Based on Petal Features', fontsize=14, fontweight='bold')
plt.legend(title='Species', fontsize=10, title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iris_petal_scatter.png', dpi=300, bbox_inches='tight')
plt.show()


# 8. Histogram i shpërndarjes së veçorive
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(feature_names):
    for species in target_names:
        species_data = df[df['species'] == species][feature]
        axes[idx].hist(species_data, bins=15, alpha=0.6,
                       label=species, color=colors[species])

    axes[idx].set_xlabel('Vlera (cm)', fontsize=10)
    axes[idx].set_ylabel('Frekuenca', fontsize=10)
    axes[idx].set_title(f'Shpërndarja: {feature}', fontsize=11, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Histogramet e shpërndarjes së veçorive', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("TË GJITHA VIZUALIZIMET JANË GJENERUAR ME SUKSES!")
print("=" * 60)
print("\nFigurat e ruajtura:")
print("  1. iris_class_distribution.png")
print("  2. iris_boxplots.png")
print("  3. iris_pairplot.png")
print("  4. iris_correlation_heatmap.png")
print("  5. iris_violin_plots.png")
print("  6. iris_petal_scatter.png")
print("  7. iris_histograms.png")