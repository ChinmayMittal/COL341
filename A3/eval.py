import csv
import traceback
data=[]
try: 
    with open('./out/test_31a.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            data.append(lines)
except: 
    traceback.print_exc()


acc=0
correct=[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 ]
try:
    for i in range(len(data)):
        assert data[i][0] == ('img_'+str(i+1))
        if int(data[i][1]) == correct[i]:
            acc+=1
except: 
    traceback.print_exc()

print(acc/len(correct))