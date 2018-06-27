
import os
import csv

reader = csv.reader(open('./BEPS.csv', 'r'), delimiter=',')
out_writer1 = csv.writer(open('./BEPS_train.csv', 'w'), delimiter=',', lineterminator='\n')
out_writer2 = csv.writer(open('./BEPS_test.csv', 'w'), delimiter=',', lineterminator='\n')
headers = next(reader)
out_writer1.writerow(headers)
out_writer2.writerow(headers)
for i, row in enumerate(reader):
    if i % 10 == 0:
        out_writer2.writerow(row)
    else:
        out_writer1.writerow(row)