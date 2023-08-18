#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

void createChildren(int n, int m) {
    if (n == 0) {
        return;
    }
    int pid_obt = fork();
    if (pid_obt == 0) {
        int child_pid = getpid();
        int parent_pid = getppid();
        printf("Child %d is created\n", child_pid);
        createChildren(n-1, m);
    }
    else {
        wait(NULL);
        int parent_pid = getpid();
        printf("Child %d with parent %d exited\n", pid_obt, parent_pid);
        if (n == m) {
            printf("Parent exited\n");
        }
    }
}

int main () {
    int n;
    scanf("%d", &n);
    int parent_pid = getpid();
    printf("Parent is : %d\n", parent_pid);
    printf("Number of children: %d\n", n);
    createChildren(n, n);
}