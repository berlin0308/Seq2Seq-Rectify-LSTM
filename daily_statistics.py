import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def calculate_statistic(df, classes=['AL','AS','DR','FD','NL','NS','RM','X'], target_col=None):

    # print(df['Date'].value_counts())
    print("Average Score:",df['Score'].mean())
    # print(behavior_counts)

    print("\nBehavior Occurrences:")    
    behavior_counts = df[target_col].value_counts()
    occur_counts = [0,0,0,0,0,0,0,0]
    for tup in behavior_counts.items():
        occur_counts[tup[0]] = tup[1]

    # print(occur_counts)
    for behavior, count in enumerate(occur_counts):
        print(f"class : {classes[behavior]}, count: {count} [{count} min] [{round(count/60,1)} hr]")


    print("\nBehavior Bouts:")
    bout_counts = [0,0,0,0,0,0,0,0]
    current_behavior = 8

    for _, row in df.iterrows():
        behavior = row[target_col]
        
        if behavior != current_behavior:
            bout_counts[behavior] += 1
            current_behavior = behavior

    # print(bout_counts)
    for behavior, count in enumerate(bout_counts):
        print(f"class : {classes[behavior]}, bout: {count}")

    print("\nAverage Duration per Bout")
    ADPBs = []
    for i in range(8):
        count = occur_counts[i]
        bouts = bout_counts[i]

        if bouts == 0:
            ADPB = 0
        else:
            ADPB = round(float(count / bouts), 2)

        ADPBs.append(ADPB)

    for behavior, count in enumerate(ADPBs):
        print(f"class : {classes[behavior]}, ADPB: {count} [{count} min]")

    return occur_counts, bout_counts, ADPBs

def write_statistics_csv(input_files, output_path):

    all_data = pd.DataFrame()

    classes = ['AL','AS','DR','FD','NL','NS','RM','X']
    columns = ['Date'] + [f'Occurs-{cls}' for cls in classes] + [f'Bouts-{cls}' for cls in classes] + [f'ADPB-{cls}' for cls in classes]
    all_data = pd.DataFrame(columns=columns)

    for file in input_files:
        df = pd.read_csv(file, delimiter=',', encoding='utf-8')
        occurs, bouts, ADPBs = calculate_statistic(df)
        data_row = [file] + occurs + bouts + ADPBs
        all_data.loc[len(all_data)] = data_row

    all_data.to_csv(output_path, index=False)

def plot_statistics_truth_predict(truth_data, predict_data):

    print(truth_data)
    print(predict_data)
    MAPE = np.mean(np.abs((np.array(truth_data) - np.array(predict_data)) / np.array(truth_data))) * 100
    print(f'MAPE: {MAPE}')

    labels = ['AL','AS','DR','FD','NL','NS','RM','X']

    # The width of the bars
    bar_width = 0.35

    # The x location for the groups
    index = np.arange(len(labels))

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    bar1 = plt.bar(index, truth_data, bar_width, label='Manual Count', color='steelblue')
    bar2 = plt.bar(index + bar_width, predict_data, bar_width, label='Model Predictions', color='lightblue')

    # Adding titles and labels
    plt.xlabel('Date (Month-day)')
    plt.ylabel('Cow face detection total times')
    plt.title('Comparison of Detection Results')
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    # Showing the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    csv_file_path = "daily_data/raw/V9_lr5e-4_ep80_best/1112.csv"
    # csv_file_path = "inf_day_2023" + date + ".csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    occurs_truth, bouts_truth, ADPBs_truth = calculate_statistic(df, target_col="Truth")
    occurs_predict, bouts_predict, ADPBs_predict = calculate_statistic(df, target_col="Predict")

    plot_statistics_truth_predict(occurs_truth, occurs_predict)
    # print(occurs, bouts, ADPBs)

    # input_files = ["daily_assess/1112.csv","daily_assess/1103.csv"]
    # write_statistics_csv(input_files, "output.csv")

""" 1112
[Truth]
Behavior Occurrences:
class : AL, count: 158 [158 min] [2.6 hr]
class : AS, count: 274 [274 min] [4.6 hr]
class : DR, count: 36 [36 min] [0.6 hr]
class : FD, count: 37 [37 min] [0.6 hr]
class : NL, count: 636 [636 min] [10.6 hr]
class : NS, count: 12 [12 min] [0.2 hr]
class : RM, count: 256 [256 min] [4.3 hr]
class : X, count: 31 [31 min] [0.5 hr]

[Predict]
Behavior Occurrences:
class : AL, count: 100 [100 min] [1.7 hr]
class : AS, count: 251 [251 min] [4.2 hr]
class : DR, count: 65 [65 min] [1.1 hr]
class : FD, count: 33 [33 min] [0.6 hr]
class : NL, count: 645 [645 min] [10.8 hr]
class : NS, count: 22 [22 min] [0.4 hr]
class : RM, count: 324 [324 min] [5.4 hr]
class : X, count: 0 [0 min] [0.0 hr]

"""