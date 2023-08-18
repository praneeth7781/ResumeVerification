#include "types.h"
#include "stat.h"
#include "user.h"

int main(int argc, char *argv[]) {
    if (fork() == 0) {
        char* args[argc];
        for (int i = 1; i < argc; i++) {
            args[i-1] = argv[i];
        }
        args[argc-1] = '\0';

        exec(argv[1], args);
    }
    else {
        wait();
    }
    exit();
}
