// Code adapted from cpp-capture example
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include <librealsense/rs.hpp>
#include "hand-capture.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include "time.h"
#include "math.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAXFRAMES 10000

texture_buffer buffers[RS_STREAM_COUNT];
bool enable_recording = false;
std::string filepath;
int i;

// Split the screen into 640X480 tiles, according to the number of supported streams. Define layout as follows : tiles -> <columds,rows>
const std::map<size_t, std::pair<int, int>> tiles_map = {       { 1,{ 1,1 } },
                                                                { 2,{ 2,1 } },
                                                                { 3,{ 2,2 } },
                                                                { 4,{ 2,2 } },
                                                                { 5,{ 3,2 } },          // E.g. five tiles, split into 3 columns by 2 rows mosaic
                                                                { 6,{ 3,2 } }};

int main(int argc, char * argv[]) try
{
    rs::log_to_console(rs::log_severity::warn);

    rs::context ctx;
    if (ctx.get_device_count() == 0) throw std::runtime_error("No device detected. Is it plugged in?");
    rs::device & dev = *ctx.get_device(0);

    std::vector<rs::stream> supported_streams;

    // Add depth and color to the supported streams list
    supported_streams.push_back((rs::stream)rs::capabilities::depth);
    supported_streams.push_back((rs::stream)rs::capabilities::color);


    // Configure all supported streams to run at 30 frames per second
    dev.enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 30, rs::output_buffer_format::continous);
    dev.enable_stream(rs::stream::depth, 480, 360, rs::format::z16, 30, rs::output_buffer_format::continous);

    // Compute field of view for each enabled stream
    for (auto & stream : supported_streams)
    {
        if (!dev.is_stream_enabled(stream)) continue;
        auto intrin = dev.get_stream_intrinsics(stream);
        std::cout << "Capturing " << stream << " at " << intrin.width << " x " << intrin.height;
        std::cout << std::setprecision(1) << std::fixed << ", fov = " << intrin.hfov() << " x " << intrin.vfov() << ", distortion = " << intrin.model() << std::endl;
    }

    // Start our device
    dev.start();

    // Open a GLFW window
    glfwInit();
    std::ostringstream ss; ss << "Hand-Capture (" << dev.get_name() << ")";

    std::string rgb_filename, z16_filename;

    int rows = tiles_map.at(supported_streams.size()).second;
    int cols = tiles_map.at(supported_streams.size()).first;
    int tile_w = 640; // pixels
    int tile_h = 480; // pixels
    GLFWwindow * win = glfwCreateWindow(tile_w*cols, tile_h*rows, ss.str().c_str(), 0, 0);
    glfwSetWindowUserPointer(win, &dev);
    glfwSetKeyCallback(win, [](GLFWwindow * win, int key, int scancode, int action, int mods)
    {
        auto dev = reinterpret_cast<rs::device *>(glfwGetWindowUserPointer(win));
        if (action != GLFW_RELEASE) switch (key)
        {
        case GLFW_KEY_E:
            if (dev->supports_option(rs::option::r200_emitter_enabled))
            {
                int value = !dev->get_option(rs::option::r200_emitter_enabled);
                std::cout << "Setting emitter to " << value << std::endl;
                dev->set_option(rs::option::r200_emitter_enabled, value);
            }
            break;
        case GLFW_KEY_A:
            if (dev->supports_option(rs::option::r200_lr_auto_exposure_enabled))
            {
                int value = !dev->get_option(rs::option::r200_lr_auto_exposure_enabled);
                std::cout << "Setting auto exposure to " << value << std::endl;
                dev->set_option(rs::option::r200_lr_auto_exposure_enabled, value);
            }
            break;
        case GLFW_KEY_R:
            enable_recording = !enable_recording;
            if (enable_recording){
                std::cout << "START RECORDING" << std::endl;
                // Reset count and create a new directory, if it doesn't exist yet
                i=1;
                filepath = std::string("./out-") + std::to_string((int)time(NULL)) + std::string("/");
                /* check if directory exist */
                struct stat status = { 0 };
                if( stat(filepath.c_str(), &status) == -1 ) {
                    /* create it */
                    mkdir( filepath.c_str(), 0700 );
                }
            }
            else {
                std::cout << "STOP RECORDING" << std::endl;
            }
        }
    });
    glfwMakeContextCurrent(win);

	// Set autoexposure to true
	if (dev.supports_option(rs::option::r200_lr_auto_exposure_enabled))
	{
		int value = !dev.get_option(rs::option::r200_lr_auto_exposure_enabled);
		std::cout << "Setting auto exposure to " << value << std::endl;
		dev.set_option(rs::option::r200_lr_auto_exposure_enabled, value);
	}

    // Loop until window is open
    while (!glfwWindowShouldClose(win))
    {
        // If MAXFRAMES limit is reached, stop recording
        if(i>=MAXFRAMES && enable_recording){
            enable_recording = false;
            std::cout << "STOP - TOO MANY FRAMES: Please press again R to start recording" << std::endl;
        }
        // Wait for new images
        glfwPollEvents();
        dev.wait_for_frames();

        // Clear the framebuffer
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw the images and eventually save them
        glPushMatrix();
        glfwGetWindowSize(win, &w, &h);
        glOrtho(0, w, h, 0, -1, +1);
        if(enable_recording) {
            // Draw the images and save them in the output directory with an incremental number filename
            rgb_filename = filepath + std::string(ceil(log10((float)MAXFRAMES/i))-1, '0') + std::to_string(i) + ".rgb";
            z16_filename = filepath + std::string(ceil(log10((float)MAXFRAMES/i))-1, '0') + std::to_string(i) + ".z16";
            buffers[0].showAndSave(dev, rs::stream::color, 0, 0, tile_w, tile_h, rgb_filename.c_str());
            buffers[1].showAndSave(dev, rs::stream::depth_aligned_to_color, w / cols, 0, tile_w, tile_h, z16_filename.c_str());
            i++;
        }
        else {
            // Just draw them
            buffers[0].show(dev, rs::stream::color, 0, 0, tile_w, tile_h);
            buffers[1].show(dev, rs::stream::depth_aligned_to_color, w / cols, 0, tile_w, tile_h);
        }
        glPopMatrix();
        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}
catch (const rs::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
