import csv

data = open('data.csv', 'w')

writer = csv.writer(data)
l=[]
for i in range(154) :
    l.append("p" + str(i + 1))

writer.writerow(l)