#include "types.h"
#include "stat.h"
#include "user.h"

char buf[512];

void
cat(int fd, int lc)
{
  int n;
  int l = 0;

  while((n = read(fd, buf, sizeof(buf))) > 0) {
    if (fd == 0) {
        if (write(1, buf, n) != n) {
            printf(1, "cat: write error\n");
            exit();
        }
    }
    else {
        int c = 0;
        for (int i = 0; i < n; i++) {
            c++;
            if (buf[i] == '\n') {
                l++;
                if (l == lc) {
                    break;
                }
            }
        }
        if (write(1, buf, c) != c) {
            printf(1, "cat: write error\n");
            exit();
        }
        else {
            break;
        }
    }
  }
  if(n < 0){
    printf(1, "cat: read error\n");
    exit();
  }
}

int
main(int argc, char *argv[])
{
  int fd, i;

  if(argc <= 1){
    cat(0, 0);
    exit();
  }

  int lc = atoi(argv[1]);
  printf(1, "Line count : %d\n", lc);
  printf(1, "Argument count : %d\n", argc);

  for(i = 2; i < argc; i++){
    if((fd = open(argv[i], 0)) < 0){
      printf(1, "head: cannot open %s\n", argv[i]);
      exit();
    }
    printf(1, "-----%s-----\n", argv[i]);
    cat(fd, lc);
    close(fd);
  }
  exit();
}
