import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.loadtxt('./data/scaled_data/X_train.txt')

correlation_matrix = np.corrcoef(data, rowvar=False)

plt.figure(figsize=(10, 8))  # Adjust size as needed

# Generate the heatmap
sns.heatmap(correlation_matrix, 
            annot=True,        # Show values in each cell
            cmap='coolwarm',   # Color map ('viridis', 'plasma', 'coolwarm', etc.)
            fmt=".2f",         # Format for annotations (2 decimal places)
            linewidths=0.5)    # Add lines between cells

plt.title('Feature Correlation Heatmap')
plt.show()
