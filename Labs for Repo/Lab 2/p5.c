#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main () {
    int p1[2];
    int p2[2];
    if (pipe(p1) < 0) {
        printf("Error");
    }
    if (pipe(p2) < 0) {
        printf("Error");
    }
    if (fork() == 0) {
        int a;
        scanf("%d", &a);
        printf("Process A : Input value of x : %d\n", a);
        write(p1[1], &a, sizeof(a)+1);
    }
    else {
        wait(NULL);
        int a;
        read(p1[0], &a, sizeof(a)+1);
        close(p1[1]);
        if (fork() == 0) {
            int b;
            scanf("%d", &b);
            printf("Process B : Input value of y : %d\n", b);

            int c = a+b;
            write(p2[1], &c, sizeof(c)+1);
        }
        else {
            wait(NULL);
            int c;
            c = read(p2[0], &c, sizeof(c)+1);
            printf("Process C : Result after addition : %d\n", c);
        }
    }
}