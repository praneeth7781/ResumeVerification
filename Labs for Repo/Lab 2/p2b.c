#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main () {
    int n;
    scanf("%d", &n);
    int pid_obt = fork();
    if (pid_obt == 0) {
        int child_pid = getpid();
        for (int i = 1; i <= n; i++) {
            printf("C %d %d\n", child_pid, i);
        }
    }
    else {
        wait(NULL);
        int parent_pid = getpid();
        for (int i = n+1; i <= 2*n; i++) {
            printf("P %d %d\n", parent_pid, i);
        }
    }
}