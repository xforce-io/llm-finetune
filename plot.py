import sys
import matplotlib.pyplot as plt
import numpy as np


kPattern = "train_loss="
kStop = "]"

if __name__ == "__main__":
    filepath = sys.argv[1]
    fp = open(filepath, "r", encoding="ISO-8859-1")
    results = []
    for line in fp:
        curIdx = 0
        nextIdx = line.find(kPattern, curIdx)
        while nextIdx >= 0:
            try:
                stopIdx = line.find(kStop, nextIdx)
                results += [float(line[nextIdx+len(kPattern):stopIdx])]
            except Exception:
                print(f"error[{line}] curIdx[{curIdx}] nextIdx[{nextIdx}]")
            curIdx = stopIdx
            nextIdx = line.find(kPattern, curIdx)

    smoothed_data = np.convolve(results, np.ones(100)/100, mode='valid')  # 使用简单移动平均进行平滑处理
    plt.plot(results, marker='o', linestyle='-', color='b', label='Data')
    plt.plot(smoothed_data, marker='', linestyle='-', color='r', label='Smoothed Data')
    plt.title('Line Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    output_file_path = 'line_chart.png'
    plt.savefig(output_file_path)
