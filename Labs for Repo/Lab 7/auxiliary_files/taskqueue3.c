#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

// Global variables
long sum = 0;
long odd = 0;
long even = 0;
long min = INT_MAX;
long max = INT_MIN;
bool done = false;
int totalBusy = 0;

int MAX_BUFFER_SIZE = 100;
int totaldone = 0;

pthread_cond_t c1, c2;
pthread_mutex_t lock;


void processtask(long number);

void processtask(long number)
{
    // simulate burst time
    sleep(number);

    // update global variables
    pthread_mutex_lock(&lock);
    sum += number;
    if (number % 2 == 1)
    {
        odd++;
    }
    else
    {
        even++;
    }
    if (number < min)
    {
        min = number;
    }
    if (number > max)
    {
        max = number;
    }
    pthread_mutex_unlock(&lock);
}

struct myargs_t{
    char* charBuffer;
    long* intBuffer;
    int* busy;
    int* valid;
    int buffersize;
    long totalProcesses;
};

void* implement(void* args) {
    struct myargs_t* arg = (struct myargs_t*)(args);
    long* intBuffer = (arg)->intBuffer;
    char* charBuffer = (arg)->charBuffer;
    int* busy = (arg)->busy;
    int* valid = (arg)->valid;
    int current;
    bool signal = false;

    while (true) {
        pthread_mutex_lock(&lock);
        if (totalBusy >= arg->totalProcesses) {
            signal = true;
            break;
        }
        while ((arg)->buffersize <= 0) {
            pthread_cond_wait(&c2, &lock);
        }
        if (totalBusy >= arg->totalProcesses) {
            signal = true;
            break;
        }
        for (int i = 0; i < MAX_BUFFER_SIZE; i++) {
            if (arg->busy[i] == 0 && arg->valid[i]) {
                arg->busy[i] = 1;
                current = i;
                totalBusy++;
                break;
            }
        }
        char type = charBuffer[current];
        int num = intBuffer[current];
        pthread_mutex_unlock(&lock);

        if (type == 'p') {
            processtask(intBuffer[current]);
            pthread_cond_signal(&c1);
            printf("Task completed\n");
        }
        else if (type == 'w') {
            // waiting period
            sleep(intBuffer[current]);
            pthread_cond_signal(&c1);
            printf("Wait Over\n");
        }

        pthread_mutex_lock(&lock);
        (arg)->buffersize--;
        printf("%d %ld\n", (arg)->buffersize, pthread_self());
        (arg)->busy[current] = 0;
        (arg)->valid[current] = 0;
        pthread_cond_signal(&c1);
        totaldone++;
        printf("Total done %d\n", totaldone);
        pthread_mutex_unlock(&lock);
    }
    if (signal) {
        pthread_mutex_unlock(&lock);
    }
}

int main(int argc, char *argv[])
{
    char* charBuffer = (char*)malloc(sizeof(char)*MAX_BUFFER_SIZE);
    long* intBuffer = (long*)malloc(sizeof(long)*MAX_BUFFER_SIZE);
    int* busy = (int*)malloc(sizeof(int)*MAX_BUFFER_SIZE);
    int* valid = (int*)malloc(sizeof(int)*MAX_BUFFER_SIZE);

    struct myargs_t* args = (struct myargs_t*)malloc(sizeof(struct myargs_t));
    args->charBuffer = charBuffer;
    args->intBuffer = intBuffer;
    args->busy = busy;
    args->valid = valid;

    for (int i = 0; i < MAX_BUFFER_SIZE; i++) {
        args->busy[i] = 0;
        args->valid[i] = 0;
    }
    args->buffersize = 0;

    if (argc != 3)
    {
        printf("Usage: sum <infile>\n");
        exit(EXIT_FAILURE);
    }
    int numThreads = atoi(argv[2]);

    char *fn = argv[1];
    // Read from file
    FILE *fin = fopen(fn, "r");
    long t;
    fscanf(fin, "%ld\n", &t);
    printf("The number of tasks are : %ld \n", t);
    args->totalProcesses = t;
    char type;
    long num;

    pthread_t th[numThreads];
    for (int i = 0; i < numThreads; i++) {
        if (pthread_create(th + i, NULL, &implement, (void *)(args)) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    while (fscanf(fin, "%c %ld\n", &type, &num) == 2) {
        pthread_mutex_lock(&lock);
        while (args->buffersize >= MAX_BUFFER_SIZE) {
            pthread_cond_wait(&c1, &lock);
        }
        for (int i = 0; i < MAX_BUFFER_SIZE; i++) {
            if (busy[i] == 0 && valid[i] == 0) {
                charBuffer[i] = type;
                intBuffer[i] = num;
                valid[i] = 1;
                break;
            }
        }
        args->buffersize++;
        printf("Hehe %d\n", args->buffersize);
        pthread_cond_signal(&c2);
        pthread_mutex_unlock(&lock); 
    }
    fclose(fin);
    for (int i = 0; i < numThreads; i++) {
        pthread_join(th[i], NULL);
    }
    // Print global variables
    printf("%ld %ld %ld %ld %ld\n", sum, odd, even, min, max);

    return (EXIT_SUCCESS);
}
