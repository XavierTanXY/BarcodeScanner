#!/bin/bash

echo "Running Task 2 ..."
echo "Setting up directories for task 2 ..."

#Make directories for task2
mkdir -p /home/student/tan_xhienyi_18249833/output/task2

echo "Setting up sub-directories for task 2 ..."
#Make sub-directories for task2
mkdir -p /home/student/tan_xhienyi_18249833/output/task2/barcode1/
mkdir -p /home/student/tan_xhienyi_18249833/output/task2/barcode2/
mkdir -p /home/student/tan_xhienyi_18249833/output/task2/barcode3/
mkdir -p /home/student/tan_xhienyi_18249833/output/task2/barcode4/
mkdir -p /home/student/tan_xhienyi_18249833/output/task2/barcode5/

#Run my program
python /home/student/tan_xhienyi_18249833/src/task2.py

echo "Done!"
