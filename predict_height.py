import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

heights = np.array([60, 65, 70, 75])
weights = np.array([110, 140, 150, 180])

heights_reshaped = heights.reshape(-1, 1)

model = LinearRegression()

model.fit(heights_reshaped, weights)

heights_predict = np.array([50, 75, 120]).reshape(-1, 1)
weights_predict = model.predict(heights_predict)

for i, height in enumerate(heights_predict):
    print(f'Predicted weight for height {height[0]} inches: {weights_predict[i]:.2f} lbs')


plt.scatter(heights, weights, label='Original data')
plt.plot(heights_predict, weights_predict, label='Regression line', color='red')

plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('Linear Regression: Height vs Weight')
plt.legend()

for i, txt in enumerate(weights_predict):
    plt.annotate(f'{weights_predict[i]:.2f} lbs', (heights_predict[i], weights_predict[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
