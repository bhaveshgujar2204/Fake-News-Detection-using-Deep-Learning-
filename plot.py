import matplotlib.pyplot as plt
import numpy as np

fruits = ['BERT Model', 'Decision Tree', 'Random Forest', 'KNN']
sales = [97, 82, 79, 73]

plt.bar(fruits, sales, width=0.5)
plt.xlabel('Algorithms')
plt.ylabel('F1-Score')
plt.show()