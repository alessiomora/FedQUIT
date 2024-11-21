#!/usr/bin/bash

iterations=1

for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m centralized_fedquit.experiment experiment="rocket"
    python -m centralized_fedquit.experiment experiment="mushroom"
    python -m centralized_fedquit.experiment experiment="baby"
    python -m centralized_fedquit.experiment experiment="lamp"
    python -m centralized_fedquit.experiment experiment="sea"
done


