# -*- coding: utf-8 -*-

fin = open("/mnt/data1/dev/data/book/book.txt", encoding="utf-8")
flines = open("/mnt/data1/dev/data/book/book_lines.txt", "w")
fout = open("/mnt/data1/dev/data/book/book_03.txt", "w")

kThrWrite = 2048
kWriteLine = True

def getSepIdx(buf, startIdx) :
    sepIdx = buf.find("。", startIdx)
    if sepIdx >= 0:
        return sepIdx
    return buf.find(".", startIdx)

def getLines():
    buf = ""
    lines = []
    line = True
    while line :
        try :
            line = fin.readline()
            if line:
                buf += line.strip()
                curIdx = 0
                sepIdx = getSepIdx(buf, curIdx)
                while sepIdx >= 0:
                    lines += [buf[curIdx:sepIdx+1].strip()]

                    curIdx = sepIdx+1
                    sepIdx = getSepIdx(buf, curIdx)
                buf = buf[curIdx:]
        except Exception as e:
            print(f"error[{e}]")
            line = "-"
    return lines

def strategyLen(lines):
    kThrLen = 20
    result = []
    for line in lines:
        if len(line) >= kThrLen:
            result += [line]
    return result

def filter(lines, strategies) :
    for strategy in strategies:
        lines = strategy(lines)
    return lines

def writeRes(lines):
    writeBuf = ""
    for line in lines:
        writeBuf += line
        if len(writeBuf) > kThrWrite:
            fout.write(writeBuf + "\n")
            writeBuf = ""

if __name__ == "__main__":
    lines = getLines()

    if kWriteLine:
        for line in lines:
            toWrite = line.strip()
            if len(toWrite) > 0:
                flines.write(toWrite + "\n")
        flines.close()

    lines = filter(lines, [strategyLen])
    writeRes(lines)