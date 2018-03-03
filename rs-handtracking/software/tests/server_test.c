// Code adapted from cpp-capture example
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include "hand-stream.hpp"
#include <sys/types.h>
#include <signal.h>

int main(int argc, char * argv[])
{
    
    int server, socket;
    struct sockaddr_in address;
    int pid;

    server = startServer(&address);
    if(!server) {
        printf("Error 1: Server launch error\n");
    }

    // Main loop
    do{
        socket = listenAndAccept(server, &address);

        printf("Connected\n");
        pid = fork();
        if(pid==0){
            while(1){
                sendFrame(socket, "a", 2);
                sendFrame(socket, "\n", 2);
            }   
        }
        else if(pid>0){
            do{
                printf("%d\n", pid);
            } while(isConnected(socket));
            closeSocket(socket);
            kill(pid, 15);
            printf("Disconnected\n");
        }
      
    } while(1);

    return EXIT_SUCCESS;
}

