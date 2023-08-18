#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>

int main () {
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    int cur_pid = getpid();
    printf("Process Id is: %d\n", cur_pid);
    while(true) {
        printf("Waiting...\n");
        sleep(3);
    }
}