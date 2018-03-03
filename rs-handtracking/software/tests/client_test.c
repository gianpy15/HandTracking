#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#define PORT 8343
  
int main(int argc, char const *argv[])
{
    struct sockaddr_in address;
    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    unsigned char buffer[1024];
	FILE* fp;
	int i, j;
	
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }
  
    memset(&serv_addr, '0', sizeof(serv_addr));
  
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
      
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
  
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }

	for(j=0; j<900; j++){
		i=0;
		fp=fopen("out.rgb", "wb");
		while(i<640*480*3){
			valread = read(sock, buffer, 1024);
			fwrite(buffer, 1, valread, fp);
			i+=valread;
		}
		fclose(fp);
		
		i=0;
		fp=fopen("out.z16", "wb");
		while(i<640*480*2){
			valread = read(sock, buffer, 1024);
			fwrite(buffer, 1, valread, fp);
			i+=valread;
		}
		fclose(fp);
	}

    return 0;
}
