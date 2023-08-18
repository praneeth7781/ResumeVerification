#include <stdio.h>
#include <unistd.h>

int main () {
    int pid_obt = fork();
    if (pid_obt == 0) {
        int child_pid = getpid();
        int parent_pid = getppid();
        printf("Child : My parent ID is:%d\n", child_pid);
        printf("Child : The parent process ID is:%d\n", parent_pid);

        sleep(5);

        int child_pid_post_sleep = getpid();
        int parent_pid_post_sleep = getppid();

        printf("Child : After sleeping for 5 seconds");
        printf("Child : My process ID is:%d\n", child_pid_post_sleep);
        printf("Child : The parent process ID is:%d\n", parent_pid_post_sleep);

    }
    else {
        int parent_pid = getpid();
        printf("Parent : My process ID is:%d\n", parent_pid);
        printf("Parent : The child process ID is:%d\n", pid_obt);
    }
}