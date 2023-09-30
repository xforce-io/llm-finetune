kLimit = 10000000

def dedupBasic(arr):
    arr = sorted(arr)
    lenArr = len(arr)
    deduped = []
    for i in range(lenArr-1) :
        if arr[i] != arr[i+1]:
            deduped.append(arr[i])
    deduped.append(arr[-1])
    return deduped

def jaccardSimilarity(sentence1, sentence2):
    # Tokenize the sentences into sets of words
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())

    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1) + len(words2) - intersection
    similarity = intersection / union

    return similarity

def dedupJaccard(sortedArr):
    lenArr = len(sortedArr)
    deduped = []
    for i in range(lenArr-1) :
        if i % 2000 :
            print("process[%d]" % i)

        similarity = jaccardSimilarity(sortedArr[i], sortedArr[i+1])
        if similarity < 0.7:
            deduped.append(sortedArr[i])
        else:
            print("filtered sent[%s|%s|%s]" % (sortedArr[i], sortedArr[i+1], similarity))
    deduped.append(sortedArr[-1])
    return deduped

def dedup(arr):
    dedup = dedupBasic(arr)
    return dedupJaccard(dedup)

def readlinesFromFile(filepath, limit):
    with open(filepath, "r") as file:
        lines = []
        for i in range(limit):
            line = file.readline()
            if not line:
                break

            lines.append(line.strip())
    return lines

def dedepFile(inputPath, outputPath):
    lines = readlinesFromFile(inputPath, kLimit)
    lines = dedup(lines)
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

    tested = dedup(tested)
    assert len(tested) == 3

    dedepFile("/tmp/input.txt", "/tmp/input.txt")
