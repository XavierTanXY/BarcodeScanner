#!/bin/bash

echo "Running Task 3 ..."
echo "Setting up directories for task 3 ..."

#Make directories for task3
mkdir -p /home/student/tan_xhienyi_18249833/output/task3

echo "Setting up sub-directories for task 3 ..."

#Make sub-directories for task3
mkdir -p /home/student/tan_xhienyi_18249833/output/task3/barcode1/
mkdir -p /home/student/tan_xhienyi_18249833/output/task3/barcode2/
mkdir -p /home/student/tan_xhienyi_18249833/output/task3/barcode3/
mkdir -p /home/student/tan_xhienyi_18249833/output/task3/barcode4/
mkdir -p /home/student/tan_xhienyi_18249833/output/task3/barcode5/

#Run my program
python /home/student/tan_xhienyi_18249833/src/task3.py

echo "Done!"
