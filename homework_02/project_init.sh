#!/bin/bash
if [ -z "$1" ]; then
    echo "No project name provided"
    exit 1
fi

name="$1"

mkdir "$name"

mkdir "$name/data"
mkdir "$name/scripts"
mkdir "$name/results"

touch "$name/data/raw_data.txt"
chmod 600 "$name/data/raw_data.txt"

script="$name/scripts/run_analysis.sh"
echo '#!/bin/bash' > "$script"
echo 'echo "Hello from $1"' >> "$script"
