#!/usr/bin/env bash
# Detecting max num threads
max_processes=1
str=$(uname)
if [[ "$str" == 'Linux' ]]; then
   max_processes=$(nproc)
elif [[ "$str" == 'Darwin' ]]; then
   max_processes=$(sysctl -n hw.ncpu)
fi

echo "Detected $max_processes cores on your machine"
echo "I'll compress using $max_processes threads"

function expand() {
    echo processing frames from $1
    unzip -oq $1
    rm -f $1
}

echo expanding files...
processes=0
for file in *.zip
do
    processes=$(($processes + 1))
    expand "$file" & # Parallel execution
    # If there are more than 4 thread, wait
    if [ "$processes" -ge ${max_processes} ]; then
        wait
        processes=0
    fi
done
wait

rm -rf __MACOSX

echo done!