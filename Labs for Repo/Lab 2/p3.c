#include <stdio.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>

int main () {
    while (1) {
        int pid_obt = fork();
        if (pid_obt == 0) {
            printf(">>> ");
            char executable[50];
            scanf("%s", executable);
            char* array[2];
            array[0] = executable;
            array[1] = NULL;
            execv(array[0], array);
        }
        else {
            wait(NULL);
        }
    }
}