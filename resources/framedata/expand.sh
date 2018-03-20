#!/usr/bin/env bash
echo expanding files...
for file in *.zip
do
    echo processing frames from ${file}
    unzip -o ${file}
done

rm -rf __MACOSX

echo done!