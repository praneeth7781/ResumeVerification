#include <stdio.h>
#include <unistd.h>

int main () {
    int pid_obt = fork();
    if (pid_obt == 0) {
        int child_pid = getpid();
        int parent_pid = getppid();
        printf("Child : My parent ID is:%d\n", child_pid);
        printf("Child : The parent process ID is:%d\n", parent_pid);

        int inp;
        scanf("%d", &inp);
    }
    else {
        sleep(60);
    }
}