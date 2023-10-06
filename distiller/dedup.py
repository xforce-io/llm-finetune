import time
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

kLimit = 10000000
kLangs = ["zh-cn"]

def filterBasic(arr):
    arr = sorted(arr)
    lenArr = len(arr)
    filtered = []
    for i in range(lenArr-1) :
        if arr[i] != arr[i+1]:
            filtered.append(arr[i])
    filtered.append(arr[-1])
    return filtered

def jaccardSimilarity(sentence1, sentence2):
    # Tokenize the sentences into sets of words
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())

    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1) + len(words2) - intersection
    return float(intersection) / union

def filterJaccard(sortedArr):
    lenArr = len(sortedArr)
    filtered = []
    for i in range(lenArr-1) :
        if i % 2000 == 0 :
            print("process[%d]" % i)

        similarity = jaccardSimilarity(sortedArr[i], sortedArr[i+1])
        if similarity < 0.7:
            filtered.append(sortedArr[i])
        else:
            print("filtered[%s] reason[jaccard|%f]" % (sortedArr[i], similarity))
    filtered.append(sortedArr[-1])
    return filtered

def filterLang(arr):
    lenArr = len(arr)
    filtered = []
    for item in arr:
        if len(item.strip()) == 0:
            continue

        try:
            lang = detect(item)
        except LangDetectException:
            print("fail detect lang for[%s]" % item)
            continue

        if lang in kLangs:
            filtered += [item]
    return filtered

def filter(arr):
    filter = filterBasic(arr)
    filter = filterLang(filter)
    return filterJaccard(filter)

def readlinesFromFile(filepath, limit):
    with open(filepath, "r") as file:
        lines = []
        for i in range(limit):
            line = file.readline()
            if not line:
                break

            lines.append(line.strip())
    return lines

def filterFile(inputPath, outputPath):
    lines = readlinesFromFile(inputPath, kLimit)
    lines = filter(lines)
    with open(outputPath, "w") as fout:
        for line in lines:
            fout.write("%s\n" % line)
        fout.close()

if __name__ == "__main__":
    tested = [
            "i love having such a beautiful little cat",
            "i get used to such a way of living, year after year",
            "i love having such a beautiful little cat",
            "The quick brown fox jumps over the lazy dog",
            "i love having such a beautiful little dog"]

    kLangs = ["en"]
    tested = filter(tested)
    assert len(tested) == 3

    kLangs = ["zh-cn"]
    t0 = time.time()
    filterFile("/mnt/data1/dev/data/common_crawl/202210/clean_docs1.txt", "/tmp/output.txt")
    t1 = time.time()
    print("all finished cost[%f]" % (t1-t0))