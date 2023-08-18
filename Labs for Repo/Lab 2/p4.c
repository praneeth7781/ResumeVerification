#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

int main () {
    char file1[50];
    char file2[50];

    int l1 = read(STDIN_FILENO, file1, sizeof(file1));
    int l2 = read(STDIN_FILENO, file2, sizeof(file2));

    file1[l1-1] = '\0'; 
    file2[l2-1] = '\0';

    char buffer[1000];
    int rfd = open(file1, O_RDONLY);
    int bytes_read = read(rfd, buffer, 1000);
    if (bytes_read >= 1) {
        buffer[bytes_read-1] = '\0';
    }

    close(rfd);
    int wfd = open(file2, O_WRONLY | O_CREAT);
    write(wfd, buffer, strlen(buffer));

    printf("%d %d %d\n", rfd, wfd, bytes_read);
    close(wfd);
}