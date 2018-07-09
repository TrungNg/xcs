#!/bin/sh

if [ ! -d ./cysrc ]; then
    mkdir cysrc
fi

for file in ./xcs_*.py
do
    echo "changing extension and moving $file"
    cp ${file##*/} ./cysrc/${file##*/}x
    echo "done ${file##*/}"
done
