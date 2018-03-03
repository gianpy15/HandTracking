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

#include <sys/types.h>
#include <signal.h>
#include <string.h>


texture_buffer buffers[RS_STREAM_COUNT];
int i;
int isShowing = 0;

// Split the screen into 640X480 tiles, according to the number of supported streams. Define layout as follows : tiles -> <columds,rows>
const std::map<size_t, std::pair<int, int>> tiles_map = {       { 1,{ 1,1 } },
                                                                { 2,{ 2,1 } },
                                                                { 3,{ 2,2 } },
                                                                { 4,{ 2,2 } },
                                                                { 5,{ 3,2 } },          // E.g. five tiles, split into 3 columns by 2 rows mosaic
                                                                { 6,{ 3,2 } }};

int main(int argc, char * argv[]) try
{
	int server, socket;
    struct sockaddr_in address;
    int pid;  

	if(argc>1 && !strcmp(argv[1], "-s"))
		isShowing = 1;

	server = startServer(&address);
    if(!server) {
        printf("Error 1: Server launch error\n");
    }

	// Main loop
	while(1){
		socket = listenAndAccept(server, &address);

        pid=fork();
		
		if(pid==0){	
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

			if(isShowing){

				// Open a GLFW window
				glfwInit();
				std::ostringstream ss; ss << "Hand-Stream (" << dev.get_name() << ")";

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
						
					// Draw and send them
					buffers[0].showAndSend(dev, rs::stream::color, 0, 0, tile_w, tile_h, socket);
					buffers[1].showAndSend(dev, rs::stream::depth_aligned_to_color, w / cols, 0, tile_w, tile_h, socket);

					glPopMatrix();
					glfwSwapBuffers(win);
				}

				glfwDestroyWindow(win);
				glfwTerminate();
				return EXIT_SUCCESS;
			}
			else {
				while(1) {
					dev.wait_for_frames();

					// Just send frames
					buffers[0].getAndSend(dev, rs::stream::color, socket);
					buffers[1].getAndSend(dev, rs::stream::depth_aligned_to_color, socket);
				}
				return EXIT_SUCCESS;
			}
		}

		else if(pid>0) {
			do{

		    } while(isConnected(socket));
		    closeSocket(socket);
		    kill(pid, 15);
		}
	}
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

