import sys
import matplotlib.pyplot as plt

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

    plt.plot(results, marker='o', linestyle='-', color='b', label='Data')
    plt.title('Line Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    output_file_path = 'line_chart.png'
    plt.savefig(output_file_path)
