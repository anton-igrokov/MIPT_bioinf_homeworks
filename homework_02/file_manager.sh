#!/bin/bash

if [ -f "report.txt" ]; then
    echo "File exists, removing..."
    rm report.txt
else
    echo "File not found, creating..."
    touch report.txt
fi
