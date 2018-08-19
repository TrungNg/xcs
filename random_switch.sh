#!/usr/bin/env bash

if grep -c "#import crandom" xcs_run.py; then
    sed -i -e 's/\#import crandom/import crandom/g' *.py
    sed -i -e 's/import random/\#import random/g' *.py
else
    sed -i -e 's/import crandom/\#import crandom/g' *.py
    sed -i -e 's/\#import random/import random/g' *.py
fi