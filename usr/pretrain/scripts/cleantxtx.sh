#!/bin/bash

# Remove all .txt files under current directory and subdirectories
find . -type f -name "*.out" -exec rm -f {} \;

echo "All .txt files have been deleted."
