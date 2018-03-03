# Hand-Capture


## Table of Contents
* [Installation Guide](#installation-guide)
* [Functionality](#functionality)
* [Compatible Devices](#compatible-devices)
* [Compatible Platforms](#compatible-platforms)
* [Usage](#usage)
* [Output](#output)
* [License](#license)

## Installation Guide

* [Linux](./doc/installation.md)

## Functionality

1. Native streams: depth, color
2. Synthetic streams: depth aligned to color
3. Stream capture: .rgb and .z16 binary files

## Compatible Devices

1. RealSense R200
2. RealSense F200
3. RealSense SR300
4. RealSense LR200
5. [RealSense ZR300](https://newsroom.intel.com/chip-shots/intel-announces-tools-realsense-technology-development/)

## Compatible Platforms

This software is written in standards-conforming C++11 and relies only on the C89 ABI for its public interface. It is developed and tested on the following platforms:

1. Ubuntu 14.04 and 16.04 LTS (GCC 4.9 toolchain)

## Usage

* 	Hand-Capture: To launch hand-capture, just type hand-capture in the terminal: color and depth-aligned-to-color streams will be displayed.
	To start recording, press R: the output will be saved in a new directory inside your actual directory, named out-TIMESTAMP, where TIMESTAMP is the time when the program was launched.
	In order to avoid recording frames accidentally, the maximum number of files inside the same directory is 9999 rgb files and 9999 z16 files.
	If the maximum number of files is reached, just press R to start a new recording. If you want to disable/enable autoexposure (enabled by default), press A;
	if you want to disable/enable IR emitter (enabled by default), press E.

*	Hand-Stream: To laucnh hand-stream, just type hand-stream in the terminal: if you want to see a preview of the streamed video, add parameter "-s" after hand-stream.
	Streaming and eventual preview will start as soon as a client connects to the server and a TCP socket is created; this application is supposed to be used on localhost only and runs on the PORT 8343.
	The streaming is a sequence of 1024 bytes TCP packets (note that the operating system may split a packet into 2 different ones): the RGB frame is sent first (3*640*480=921600 bytes), 
	then the Z16 frame is sent (2*640*480=614400 bytes), then the following RGB frame and so on. Autoexposure and IR emitter are enabled by default; in the preview mode they can be disabled/enabled
	by pressing A and E respectively. In the preview-less mode, it is not possible to change autoexposure and emitter behavior.

## Output

*	RGB: RGB frames have .rgb extension; each pixel is composed by 3 bytes (one for each color), the width of the image is 640 pixels and the height is 480 pixels.
*	Z16: depth frames have .z16 extension; each pixel is composed by 2 bytes, which represent the depth of the corresponding RGB pixel; image resolution is 640x480 pixels.

## License

This softare uses IntelRealSense/librealsense APIs. The code of this software is adapted from the source code of cpp-capture example.

https://github.com/IntelRealSense/librealsense/tree/v1.12.1

Copyright 2016 Intel Corporation
