#!/usr/bin/env bash

# Detecting max num threads
max_processes=1
str=`uname`
if [[ "$str" == 'Linux' ]]; then
   max_processes=$(nproc)
elif [[ "$str" == 'Darwin' ]]; then
   max_processes=$(sysctl -n hw.ncpu)
fi

echo "Detected $max_processes cores on your machine"
echo "I'll compress using $max_processes threads"

echo compressing files...
processes=0
for file in hands*
do
    processes=$(($processes + 1))
    echo processing frames from ${file}
    zip -rq ${file%.*}.zip ${file} & # Parallel execution
    # If there are more than 4 thread, wait
    if [ "$processes" -ge ${max_processes} ]; then
        wait
        processes=0
    fi
done
wait

echo compress finished!
echo creating archive...

zip -q framedata.zip *.zip

echo archive created!

echo deleting all zips...

rm hand*.zip

echo done!
