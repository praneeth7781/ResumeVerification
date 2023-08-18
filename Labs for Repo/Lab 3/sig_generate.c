#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

int main (int argc, char* argv[]) {
    int pid = atoi(argv[1]);
    
    printf("SIGINT signal sent to PID: %d\n", pid);
    kill(pid, SIGINT);

    printf("SIGTERM signal sent to PID: %d\n", pid);
    kill(pid, SIGTERM);

    printf("SIGKILL signal sent to PID: %d\n", pid);
    kill(pid, SIGKILL);
}