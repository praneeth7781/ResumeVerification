#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#define MAX_INPUT_SIZE 1024
#define MAX_TOKEN_SIZE 64
#define MAX_NUM_TOKENS 64

// TODO: YOUR CODE HERE
// INITIALIZE DATA STRUCTURE TO STORE PIDS OF PROCESSES TO KILL LATER

/* Splits the string by space and returns the array of tokens
*
*/
char **tokenize(char *line) {

    // tokenizer variables
	char **tokens = (char **)malloc(MAX_NUM_TOKENS * sizeof(char *));
	char *token = (char *)malloc(MAX_TOKEN_SIZE * sizeof(char));
	int i, tokenIndex = 0, tokenNo = 0;

    // loop on length of line
	for(i=0; i < strlen(line); i++){

		char readChar = line[i];

        // tokenize on any kind of space
		if (readChar == ' ' || readChar == '\n' || readChar == '\t'){
			token[tokenIndex] = '\0';
			if (tokenIndex != 0) {
				tokens[tokenNo] = (char*)malloc(MAX_TOKEN_SIZE*sizeof(char));
				strcpy(tokens[tokenNo++], token);
				tokenIndex = 0; 
			}
		}
		else {
			token[tokenIndex++] = readChar;
		}
	}
	
	free(token);
	tokens[tokenNo] = NULL ;
	return tokens;
}

// TODO
// MAKE FUNCTIONS AS REQUIRED TO AVOID REPETITIVE WORK

int main(int argc, char* argv[]) {

	char  line[MAX_INPUT_SIZE];            
	char  **tokens;           

	// TODO: YOUR CODE HERE
	// INITIALIZE GLOBAL DATA STRUCTURE OF PIDS TO SOME DEFAULT VALUE  
	// INITIALIZE OTHER VARIABLES AS NEEDED

	while(1) {	

		/* BEGIN: TAKING INPUT */
		bzero(line, sizeof(line));
		printf("$ ");
		scanf("%[^\n]", line);
		getchar();
		/* END: TAKING INPUT */

		line[strlen(line)] = '\n'; //terminate with new line
		tokens = tokenize(line);

		// You can maintain boolean variables to check which type of command is given by user and then
        // conditionally execute depending on them

        // TODO: YOUR CODE HERE


		char seriesInd[] = "&&";
		char parallelInd[] = "&&&";

		int tot = 0;
		int seriesCount = 0;
		int parallelCount = 0;


		for (int i = 0; tokens[i] != NULL; i++) {
			if (strcmp(tokens[i], seriesInd) == 0) {
				seriesCount++;
			}
			if (strcmp(tokens[i], parallelInd) == 0) {
				parallelCount++;
			}
			tot++;
		}


		if (seriesCount == 0 && parallelCount == 0) {
			char chdircmd[] = "cd";
			char errormsg[] = "bash : cd : ";
			if (strcmp(tokens[0], chdircmd) == 0) {
				if (tot != 2) {
					printf("bash : cd : too many arguments");
				}
				else {
					if (chdir(tokens[1]) != 0) {
						strcat(errormsg, tokens[1]);
						perror(errormsg);
					}
				}
			}
			else {
				char execpath[] = "/bin/";
				strcat(execpath, tokens[0]);
				if (fork() == 0) {
					execv(execpath, tokens);
				}
				else {
					wait(NULL);
				}
			}
		}
		else if (seriesCount > 0) {
			int seriesIndices[seriesCount + 2];
			int track = 1;
			seriesIndices[0] = -1;
			seriesIndices[seriesCount + 1] = tot;
			for (int i = 0; tokens[i] != NULL; i++) {
				if (strcmp(tokens[i], seriesInd) == 0) {
					seriesIndices[track] = i;
					track++;
				}
			}
			for (int i = 0; i < seriesCount + 1; i++) {
				int start = seriesIndices[i] + 1;
				int end = seriesIndices[i+1] - 1;
				char** subTokens = (char**)malloc((end-start+1)*sizeof(char*));
				for (int j = 0; j < end - start + 1; j++) {
					subTokens[j] = tokens[start + j];
				}
				
				char chdircmd[] = "cd";
				char errormsg[] = "bash : cd : ";
				if (strcmp(chdircmd, subTokens[0]) == 0) {
					if (end - start != 1) {
						printf("bash : cd : too many arguments");
					}
					else {
						if (chdir(subTokens[1]) != 0) {
							strcat(errormsg, subTokens[1]);
							perror(errormsg);
						}
					}
				}
				else {
					char execpat[] = "/bin/";
					strcat(execpat, subTokens[0]);
					if (fork() == 0) {
						execv(execpat, subTokens);
					}
					else {
						wait(NULL);
					}
				}
				free(subTokens);
			}
		}
		// else if (parallelCount > 0) {

		// }

        // TODO: YOUR CODE HERE
    
        // freeing the memory
		for(int i = 0; tokens[i] != NULL; i++){
			free(tokens[i]);
		}

		free(tokens);

	}
	return 0;
}


