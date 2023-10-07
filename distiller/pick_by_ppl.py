import os
import numpy as np

kRangePercentile = (40, 70)

kDataPath = "/mnt/data1/dev/github/distiller/data/"
kDataPrefix = "result"
kOutputPath = "output/ranged.txt"

def getFps():
    files = os.listdir(kDataPath)
    fps = []
    for filename in files:
        if filename.startswith(kDataPrefix):
            filepath = os.path.join(kDataPath, filename)
            fps += [open(filepath, "r")]
            print("file[%s] found" % filepath)
    return fps

def getPplRange(fps):
    fp = fps[0]
    fp.seek(0)

    line = fp.readline()
    ppls = []
    while line:
        items = line.split("\t")
        ppls += [float(items[0])]

        line = fp.readline()

    ppls.sort()
    return (np.percentile(ppls, kRangePercentile[0]), np.percentile(ppls, kRangePercentile[1]))

def genOutput(fps, pplRange):
    fout = open(kOutputPath, "w")
    for fp in fps:
        fp.seek(0)

        line = fp.readline()
        while line:
            items = line.strip().split("\t")

            ppl = float(items[0])
            if ppl > pplRange[0] and ppl < pplRange[1]:
                fout.write(items[1] + "\n")

            line = fp.readline()
    fout.close()

if __name__ == "__main__":
    fps = getFps()
    pplRange = getPplRange(fps)
    print("ppl range[%f - %f]" % (pplRange[0], pplRange[1]))
    genOutput(fps, pplRange)