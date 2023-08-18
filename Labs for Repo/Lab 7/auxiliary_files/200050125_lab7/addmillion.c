#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>

int account_balance = 0;
pthread_mutex_t lock;

void* increment(void* args) {
    int* arg = (int*) args;
    int million = *arg;
    pthread_mutex_lock(&lock);
    for (int j = 0; j < million; j++) {
        for (int i = 0; i < 1000000; i++) {
            account_balance++;
        }
    }
    pthread_mutex_unlock(&lock);
}

int main(int argc, char* argv[]) {
    int start = clock();
    int threadNum = atoi(argv[1]);
    pthread_t th[threadNum];
    int i;
    int millions = 2048/threadNum;
    for (i = 0; i < threadNum; i++) {
        if (pthread_create(th + i, NULL, &increment, (void *)(&(millions))) != 0) {
            perror("Failed to create thread");
            return 1;
        }
        printf("Transaction %d has started\n", i);
    }
    for (i = 0; i < threadNum; i++) {
        if (pthread_join(th[i], NULL) != 0) {
            return 2;
        }
        printf("Transaction %d has finished\n", i);
    }
    printf("Account Balance is : %d\n", account_balance);
    int end = clock();
    printf("Time spent: %f ms\n", (end - start)/(CLOCKS_PER_SEC/1000.0));
    return 0;
}