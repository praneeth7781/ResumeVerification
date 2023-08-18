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

int ind = 0;

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
    long totalProcesses;
};

void* implement(void* args) {
    struct myargs_t* arg = (struct myargs_t*)(args);
    long* intBuffer = (arg)->intBuffer;
    char* charBuffer = (arg)->charBuffer;
    int current;
    int num;
    char type;

    while (true) {
        pthread_mutex_lock(&lock);
        if (ind >= (arg->totalProcesses)) {
            pthread_mutex_unlock(&lock);
            break;
        }
        current = ind;
        num = intBuffer[ind];
        type = charBuffer[ind];
        ind++;
        pthread_mutex_unlock(&lock);

        if (type == 'p') {
            processtask(num);
            printf("Task completed\n");
        }
        else if (type == 'w') {
            // waiting period
            sleep(num);
            printf("Wait Over\n");
        }

    }
}

int main(int argc, char *argv[])
{
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

    struct myargs_t* args = (struct myargs_t*)malloc(sizeof(struct myargs_t));
    args->charBuffer = (char*)malloc(sizeof(char)*t);
    args->intBuffer = (long*)malloc(sizeof(long)*t);
    args->totalProcesses = t;

    char type;
    long num;

    int count = 0;
    while (fscanf(fin, "%c %ld\n", &type, &num) == 2) {
        args->intBuffer[count] = num;
        args->charBuffer[count] = type;
        count++;
    }
    fclose(fin);
    pthread_t th[numThreads];
    for (int i = 0; i < numThreads; i++) {
        if (pthread_create(th + i, NULL, &implement, (void *)(args)) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(th[i], NULL);
    }
    // Print global variables
    printf("%ld %ld %ld %ld %ld\n", sum, odd, even, min, max);

    return (EXIT_SUCCESS);
}
