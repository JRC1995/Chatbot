import os
import pickle
import collections

def process():
    def readAllfromFile(fileLocation, ConvDict):
        with open(fileLocation, 'rb') as file:
            lines = file.readlines()

        lines.append(b'- - dummy')
        for i in range(4, len(lines)):
            if b'- - ' in lines[i]:
                key = lines[i].decode('utf-8')
                key = key.replace('- - ', '')
                key = key.replace('\n', '')
                values = []
                if key in ConvDict:
                    values = ConvDict[key]
                for j in range(i+1, len(lines)):
                    # print('i',i,'j',j)
                    if b'- - ' in lines[j]:
                        ConvDict[key] = values

                        i = j-1
                        break
                    elif b'  - ' in lines[j]:
                        val = lines[j].decode('utf-8')
                        val = val.replace('  - ', '')
                        val = val.replace('\n', '')
                        for k in range(j+1, len(lines)):
                            if b'  - ' in lines[k] or b'- - ' in lines[k]:
                                j = k-1
                                break
                            valRest = lines[k].decode('utf-8')
                            valRest = valRest.replace('   ', '')
                            valRest = valRest.replace('\n', '')
                            val += valRest
                        values.append(val)


    ConvDict = collections.OrderedDict()


    def readFromAllfiles(folderLocation, ConvDict):
        for root, dirs, files in os.walk(folderLocation):
            for file in files:
                path = os.path.join(root, file)
                # print(path)
                readAllfromFile(path, ConvDict)


    readFromAllfiles("Chatterbot_Corpus/", ConvDict)

    #print(ConvDict)

    with open('Processed_Scripts/Chatterbot.pkl', 'wb') as fp:
        pickle.dump(ConvDict, fp)
