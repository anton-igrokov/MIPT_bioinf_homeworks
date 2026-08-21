#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Error: Not enough arguments provided"
    exit 1
fi

first_name="$1"
last_name="$2"
group="$3"

echo "Welcome, $first_name $last_name from group $group!"
