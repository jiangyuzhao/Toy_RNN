import os
import sys
import random

class SeqDataGenerator:
    def __init__(self):
        self.wordTable = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def generateSeqFile(self):
        workingDir = self.getWorkingDir()
        sourceDataFileName = os.path.join(workingDir, "dataSource")
        targetDataFileName = os.path.join(workingDir, "dataTarget")
        with open(sourceDataFileName, "w") as sourceDataFile:
            with open(targetDataFileName, 'w') as targetDataFile:
                for _ in range(1000):
                    seqList = self.generateSeq()
                    sourceDataFile.write(seqList[0] + "\n")
                    targetDataFile.write(seqList[1] + "\n")
        return

    def getWorkingDir(self):
        index = str(sys.argv[0]).rfind('/')
        return str(sys.argv[0])[:index]

    def generateSeq(self):
        seqLength = random.randint(5, 10)
        sourceStr = ""
        targetStr = ""
        for i in range(seqLength):
            if i == 0:
                index = random.randint(0, 25)
                sourceStr += self.wordTable[index]
                targetStr += self.wordTable[(index + 1) % 26]
            else:
                index = random.randint(0, 25)
                sourceStr += " " + self.wordTable[index]
                targetStr += " " + self.wordTable[(index + 1) % 26]
        return [sourceStr, targetStr]

if __name__ ==  '__main__':
    seqDataGenerator = SeqDataGenerator()
    seqDataGenerator.generateSeqFile()