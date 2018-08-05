#!/bin/sh
if [ ! -d ./cysrc ]; then
    mkdir cysrc
fi

for file in ./xcs_*.py
do
    echo "changing extension and copying $file"
    cp ${file##*/} ./cysrc/${file##*/}x
    #cp ${file##*/} ./cysrc/${file%.*}.pxd
    echo "done ${file##*/}"
done

cp crandom.py cysrc/crandom.pyx
cp rand.c cysrc/rand.c
cp rand.h cysrc/rand.h

#~/my_virtualenv/cython/bin/python3.6 setup.py build_ext --inplace
python3.6 setup.py build_ext --inplace