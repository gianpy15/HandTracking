#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>

#define PORT 8343
#define PKTSZ 1024

int startServer(struct sockaddr_in* address) {
	int server_fd;
	int opt = 1;
    int addrlen = sizeof(*address);

	if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        return 0;
    }
      
    // Forcefully attaching socket to the port 8343
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        return 0;
    }
    address->sin_family = AF_INET;
    address->sin_addr.s_addr = INADDR_ANY;
    address->sin_port = htons(PORT);
      
    if (bind(server_fd, (struct sockaddr *)address, sizeof(*address))<0)
    {
        perror("bind failed");
        return 0;
    }

    return server_fd;
}

int listenAndAccept(int server_fd, struct sockaddr_in* address) {
	int addrlen = sizeof(*address);

	if (listen(server_fd, 0) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    return accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
    
}

int sendFrame(int socket, const void* frame, int frameSize) {
	int i;

	for(i=0; i*PKTSZ<frameSize; i++){
		if(send(socket, frame+i*PKTSZ, PKTSZ, 0)<0){
			printf("error 0\n");
			return 0;
		}
	}
	return 1;
}

int isConnected(int socket) {
	char buff[1024];
	if(read(socket, buff, 1024)>0)
		return 1;
	return 0;
}

void closeSocket(int socket) {
	shutdown(socket, 2);
	close(socket);
	return;
}
