import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_truth_predict(df, date, classes=['Active\nLying','Active\nStanding','Drinking','Feeding','Non-active\nLying','Non-active\nStanding','Ruminating','X']):

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H-%M-%S", errors='coerce')

    total_duration = pd.Timedelta(hours=24)
    time_interval = total_duration / (len(df) - 1)
    df["Timestamp"] = pd.date_range(start=df["Timestamp"].min(), periods=len(df), freq=time_interval)
    
    print(df["Timestamp"].min(),df["Timestamp"].max())

    df["Timestamp"] = df["Timestamp"].dt.strftime("%H:%M:%S")
    df.set_index("Timestamp", inplace=True)

    y = df["Predict"]
    z = df["Truth"]

    plt.figure(figsize=(32, 4))

    plt.plot(y, marker='o', linestyle='', markersize=4, color='red', fillstyle='none')
    plt.plot(z, marker='o', linestyle='', markersize=2, color='black')
    # plt.plot(w, marker='o', linestyle='', markersize=4, color='blue', fillstyle='none')
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted and Actual Behavior")
    plt.title(f"Daily Assessment on {date}")
    plt.grid(True)
    x_ticks = [f"{i:02d}:00" for i in range(0, 25, 3)]  # 0, 3, 6, 9, 12, 15, 18, 21, 24
    x_positions = np.linspace(0, len(df) - 1, len(x_ticks))
    plt.xticks(x_positions, x_ticks, rotation=0)
    plt.yticks(range(8), classes)
    

    # w = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 6, 6, 6, 6, 6, 4, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 6, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 5, 5, 1, 1, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 1, 1, 6, 1, 2, 1, 3, 3, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 5, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 5, 1, 2, 3, 1, 1, 2, 2, 3, 2, 2, 1, 2, 5, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 5, 0, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 6, 0, 0, 0, 0, 4, 0, 2, 5, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 6, 0, 6, 6, 6, 6, 6, 6, 0, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 6, 4, 0, 5, 5, 2, 1, 3, 5, 1, 5, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 0, 0, 1, 1, 2, 1, 1, 1, 1, 3, 3, 1, 2, 1, 1, 5, 1, 1, 0, 6, 6, 0, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 2, 2, 6, 6, 6, 4, 6, 0, 6, 4, 4, 4, 6, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 6, 4, 4, 4, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 4, 6, 6, 1, 1, 1, 5, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 6, 6, 0, 4, 6, 4, 6, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 0, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 0, 4, 4, 4, 4, 1, 1, 5, 1, 1, 5, 1, 2, 2, 1, 1, 2, 3, 2, 1, 1, 1, 3, 1, 1, 1, 1, 5, 6, 1, 1, 1, 1, 1, 2, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 6, 0, 0, 0, 2, 6, 5, 2, 2, 2, 1, 1, 1, 0, 1, 6, 1, 3, 2, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 4, 4, 6, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 0, 6, 2, 1, 1, 2, 1, 5, 2, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 6, 0, 0, 6, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 1, 1, 2, 1, 2, 1, 1, 1, 5, 1, 2, 5, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 6, 1, 3, 1, 3, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 0, 6, 4, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 3, 3, 3, 1, 1, 3, 1, 2, 3, 1, 1, 1, 6, 6, 4, 6, 6, 4, 4, 4, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 6, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 6, 4, 4, 4, 4, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 0, 4, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    # refined_points = [i for i in range(len(y)) if y[i] != w[i]]
    # print(f'Refined Points:{len(refined_points)}')
    
    # plt.plot(refined_points, [w[i] for i in refined_points], marker='^', linestyle='', markersize=8, color='blue', fillstyle='none')
    # plt.plot(refined_points, [y[i] for i in refined_points], marker='^', linestyle='', markersize=8, color='red', fillstyle='none')
    
    # refined_points_correct = [i for i in range(len(y)) if (y[i] != w[i] and w[i] == z[i])]
    # print(f'Refined Points incorrect -> correct :{len(refined_points_correct)}')
    
    # refined_points_correct2incorrect = [i for i in range(len(y)) if (y[i] != w[i] and y[i] == z[i])]
    # print(f'Refined Points correct -> incorrect:{len(refined_points_correct2incorrect)}')
    
    # refined_points_both_incorrect = [i for i in range(len(y)) if (y[i] != w[i] and y[i] != z[i] and w[i] != z[i])]
    # print(f'Refined Points both incorrect:{len(refined_points_both_incorrect)}')


    plt.savefig(f"plots/V9_{date}_predict_truth.png")
    plt.show()


def plot_softmax_probs(df, date, classes=['Active\nLying','Active\nStanding','Drinking','Feeding','Non-active\nLying','Non-active\nStanding','Ruminating','X']):

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H-%M-%S", errors='coerce')
    total_duration = pd.Timedelta(hours=24)
    time_interval = total_duration / (len(df) - 1)
    df["Timestamp"] = pd.date_range(start=df["Timestamp"].min(), periods=len(df), freq=time_interval)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%H:%M:%S")
    df.set_index("Timestamp", inplace=True)

    # Define a color map for the lines
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    # Plotting
    plt.figure(figsize=(32, 4))

    # Plot each x0 to x6 with consistent colors
    for i in range(7):
        plt.plot(df[f'x{i}'], label=classes[i], color=colors[i])

    # Plotting the maximum value at each timestamp with the corresponding color
    for i, (_, row) in enumerate(df.iterrows()):
        x_values = row[['x0','x1','x2','x3','x4','x5','x6']]
        x_max = x_values.max()
        x_max_label = x_values.idxmax()
        color_index = int(x_max_label[1]) # Extracting the index from the label 'xi'
        plt.plot(i, x_max, marker='o', linestyle='', markersize=4, color=colors[color_index])

    plt.xlabel('Time')
    plt.ylabel('Probabilities')
    plt.title(f'Softmax Outputs on {date}')
    plt.legend(loc='lower right')

    # Adjusting x-ticks
    x_ticks = [f"{i:02d}:00" for i in range(0, 25, 3)]  # 0, 3, 6, 9, 12, 15, 18, 21, 24
    x_positions = np.linspace(0, len(df) - 1, len(x_ticks))
    plt.xticks(x_positions, x_ticks, rotation=0)
    plt.xlim(0, 1500)

    plt.tight_layout()
    plt.savefig(f"plots/V9_{date}_probs.png")

    plt.show()


if __name__ == "__main__":

    date = '1213'
    csv_file_path = f"daily_data/raw/V9_lr5e-4_ep80_best/{date}.csv"
    # csv_file_path = "inf_day_2023" + date + ".csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    print(df.info)
    plot_truth_predict(df, date)
    # plot_softmax_probs(df, date)
