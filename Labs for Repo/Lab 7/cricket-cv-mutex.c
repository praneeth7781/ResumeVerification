#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

pthread_mutex_t capitalLock, titanLock;
pthread_cond_t capitalCondition, titanCondition;

int capitalScore;
int titanScore;

int capitalCount;
int titanCount;

int* capitalScores;
int* titanScores;

void* capitalFunc(void* args) {
    int score = rand()%100;

    pthread_mutex_lock(&capitalLock);
    capitalScores[capitalCount] = score;
    capitalScore += score;
    capitalCount++;
    if (capitalCount == 11) {
        pthread_cond_signal(&titanCondition);
    }
    pthread_mutex_unlock(&capitalLock);
    return args;
}

void* titanFunc(void* args) {
    int score = rand()%100;
    pthread_mutex_lock(&titanLock);
    if (capitalCount < 11) {
        pthread_cond_wait(&titanCondition, &titanLock);
    }
    titanScores[titanCount] += score;
    titanScore += score;
    titanCount++;
    pthread_mutex_unlock(&titanLock);
    return args;
}

struct args{
    int totalScore;
    int count;
    int* scores;
};

int main (int argc, char* argv[]) {
    int totalPlayers = atoi(argv[1]);
    int numMatches = totalPlayers/22;
    
    capitalScores = (int *)malloc(11*sizeof(int));
    titanScores = (int *)malloc(11*sizeof(int));

    for (int i = 0; i < numMatches; i++) {
        for (int j = 0; j < 11; j++) {
            titanScores[j] = 0; capitalScores[j] = 0;
        }

        capitalCount = 0; titanCount = 0;
        capitalScore = 0; titanScore = 0;

        pthread_t Capitals[11];
        pthread_t Titans[11];

        for (int j = 0; j < 11; j++) {
            pthread_create(Capitals + j, NULL, (&capitalFunc), (void*)(&capitalScore));
        }
        for (int j = 0; j < 11; j++) {
            pthread_create(Titans + j, NULL, (&titanFunc), (void*)(&titanScore));
        }

        for (int j = 0; j < 11; j++) {
            pthread_join(Capitals[j], NULL);
        }
        for (int j = 0; j < 11; j++) {
            pthread_join(Titans[j], NULL);
        }
        for (int j = 0; j < 11; j++) {
            printf("%d ", capitalScores[j]);
        }
        printf("\n");
        for (int j = 0; j < 11; j++) {
            printf("%d ", titanScores[j]);
        }
        printf("\n");
    }

}