#!/usr/bin/env bash

echo compressing files...
for file in hands*
do
    echo processing frames from ${file}
    zip -rq ${file%.*}.zip ${file}
done

echo compress finished!
echo creating archive...

zip -q framedata.zip *.zip

echo archive created!

echo deleting all zips...

rm hand*.zip

echo done!
