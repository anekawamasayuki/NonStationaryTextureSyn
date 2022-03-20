#!/bin/bash
local phase=train
mkdir -p "./datasets/all/$phase"
for f in `find ./datasets/half/ -regex ".*$phase.*jpg"`; do
    echo f is $f
    newF="./datasets/all/$phase/${f##*/}"
    echo "newF is $newF"
    cp $f $newF
done
phase=test
mkdir -p "./datasets/all/$phase"
for f in `find ./datasets/half/ -regex ".*$phase.*jpg"`; do
    echo f is $f
    newF="./datasets/all/$phase/${f##*/}"
    echo "newF is $newF"
    cp $f $newF
done
