#!/bin/sh

for file in ./xcs_*.py
do
        echo "changing extension and moving $file"
        cp ${file##*/} ./cysrc/${file##*/}x
        echo "done ${file##*/}"
done
