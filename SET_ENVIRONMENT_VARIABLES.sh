#!/bin/sh
# Detect the server and set the PYTHONPATH accordingly

if [[ $(hostname) == "server" ]]; then
    export UNIMTS_ROOT="/home/focs/experiments/previous_works/simCLR/"
    export UNIMTS_DATA_ROOT="/home/focs/experiments/data/FranciscoCalatrava/"
elif [[ $(hostname) == alvis* ]]; then
    export UNIMTS_ROOT="/cephyr/users/fracal/Alvis/UniMTS/"
    export UNIMTS_DATA_ROOT="/mimer/NOBACKUP/groups/focs/"
else
    export UNIMTS_ROOT="mnt/Alvis/UniMTS/"
    export UNIMTS_DATA_ROOT"/mnt/datasets/simCLR/"
fi