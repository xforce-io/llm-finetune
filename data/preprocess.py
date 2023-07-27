# -*- coding: utf-8 -*-

import re

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

def calculateCharPercentage(input_string):
    total_chars = len(input_string)
    english_chars = sum(1 for char in input_string if char.isalpha() and ord(char) < 128)
    chinese_chars = sum(1 for char in input_string if '\u4e00' <= char <= '\u9fff')
    total_percentage = ((english_chars + chinese_chars) / total_chars) * 100
    return total_percentage

def strategyCharRatio(lines):
    kThrCharRatio = 0.4
    result = []
    for line in lines:
        if calculateCharPercentage(line) > kThrCharRatio:
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