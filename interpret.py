import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
from tqdm import tqdm

from config import OUTPUT_CSV

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=TRAIN_PATH,
                        help="File path to the CSV file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_VIS,
                        help="Directory where to save outputs.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    return args


class Interpreter():
    def __init__(self, csv_file, output_csv = OUTPUT_CSV):
        self.csv_file = csv_file
        self.output_csv = output_csv
    
    def get_data(self):
        data = pd.DataFrame(self.csv_file)
        data["all_percent"] = (data["ggo_vol"] + data["cons_vol"]) / data["lung_vol"]
        data["ggo_percent"] = data["ggo_vol"] / data["lung_vol"]
        data["cons_percent"] = data["cons_vol"] / data["lung_vol"]
        dataA = data[data["label"] == "A"]
        dataB = data[data["label"] == "B"]
        dataC = data[data["label"] == "C"]

        return dataA, dataB, dataC
    
    def calculate_thresholds(self, epsilon=1e-7):
        dataA, dataB, dataC = self.get_data()

        num_A = len(dataA)
        num_B = len(dataB)
        num_C = len(dataC)

        maximum = 0.0
        thresholds = dict()

        for i in tqdm(range(0,1000,1)):        
            for j in range(i,1000, 1):
                temp = float(i) / 1000
                j = float(j) / 1000
                percA = float(len(dataA[dataA.all_percent < temp]))
                percB = float(len(dataB[(dataB.all_percent >= temp) & (dataB.all_percent < j)]))
                percC = float(len(dataC[dataC.all_percent >= j]))

                if percA != 0.0:
                    percA = percA / num_A
                if percB != 0.0:
                    percB = percB / num_B
                if percC != 0.0:
                    percC = percC / num_C

                total = percA + percB + percC
                if total > maximum:
                    thresholds["AB"] = temp
                    thresholds["BC"] = j
                    thresholds["maximum"] = total / 3
                    maximum = total

        return thresholds

    def plot(self):
        combined_df = self.get_data()
        thresholds = self.calculate_thresholds(combined_df)
        sns.violinplot(x="all_percent",y="label", data=combined_df, split=True, linewidth=1)
        # Prvo je treba izračunat thresholde s calculate_thresholds()
        plt.axvline(thresholds["AB"]) # AB diskriminacija
        plt.axvline(thresholds["BC"]) # AC diskriminacija
        print("Total discriminative power: ", thresholds["maximum"])
        print(thresholds)

    def output(self):
        combined_df = self.get_data()
        thresholds = self.calculate_thresholds(combined_df)
        
        def toabc(x):
            if x < thresholds["AB"]: return 'A'
            if x >= thresholds["AB"] and x < thresholds["BC"]: return 'B'
            return 'C'

        combined_df["class"] = combined_df["all_percent"].apply(lambda x: toabc(x))
        combined_df[["filename_img", "class"]].to_csv(self.output_csv, index=False)


def main(args = sys.argv[1:]):
    args = parseArguments()
    interpreter = Interpreter(args.path, args.output_dir)
    interpreter.output()

if name == "__main__":
    main()
