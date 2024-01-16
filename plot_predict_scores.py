import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_paths = ['daily_data/raw/V9_lr5e-4_ep80_best/1112.csv', 'daily_data/raw/V9_lr5e-4_ep80_best/1103.csv'
              , 'daily_data/raw/V9_lr5e-4_ep80_best/1212.csv', 'daily_data/raw/V9_lr5e-4_ep80_best/1213.csv']


data = None
for file_path in file_paths:
    file_data = pd.read_csv(file_path)
    data = pd.concat([data, file_data], ignore_index=True)


classes=['Active Lying','Active Standing','Drinking','Feeding','Non-active Lying','Non-active Standing','Ruminating','X']
fig, axes = plt.subplots(7, 1, figsize=(8, 10))

for i in range(7):
    # Filter the data where Truth is i
    data_truth_i = data[data['Truth'] == i]

    # Separate the data into two groups based on the value of 'Predict'
    predict_i = data_truth_i[data_truth_i['Predict'] == i]
    predict_not_i = data_truth_i[data_truth_i['Predict'] != i]

    # Plotting for each subplot
    axes[i].hist(predict_i['Score'], bins=20, color='black', alpha=0.7, label='True')
    axes[i].hist(predict_not_i['Score'], bins=20, color='red', alpha=0.7, label='False')

    axes[i].set_title(f'{classes[i]}')
    if i == 6:
        axes[i].set_xlabel('Softmax output probabilities')
    axes[i].set_ylabel('Samples')
    axes[i].set_xlim(0, 1)
    axes[i].legend()

    x_thresh = [0.99, 0.99, 0.905, 0.99, 0.85, 0.9, 0.99]

    axes[i].axvline(x=x_thresh[i], color='blue', linestyle='--', linewidth=2)





# Adjust the layout to prevent overlap
plt.tight_layout()
plt.savefig("softmax_distribution.png")
plt.show()
