#!/usr/bin/bash

iterations=1

for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m can_bad_teaching.experiment experiment="rocket"
    python -m can_bad_teaching.experiment experiment="mushroom"
    python -m can_bad_teaching.experiment experiment="baby"
    python -m can_bad_teaching.experiment experiment="lamp"
    python -m can_bad_teaching.experiment experiment="sea"
done


