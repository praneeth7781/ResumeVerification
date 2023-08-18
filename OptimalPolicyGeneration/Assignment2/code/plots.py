import numpy as np
import matplotlib.pyplot as plt

def getValue(policy, transitionMatrix, rewardMatrix, discount, considerStates):
    '''
    First construct the coesfficient matrix. Coefficient matrix is of size SxS
    RHS will be of size (S, 1)
    '''
    coeffMatrix = np.zeros((S, S), dtype = np.float64)
    RHS = np.zeros((S, 1), dtype = np.float64)
    for i in range(S):
        coeffMatrix[i] = -discount*transitionMatrix[policy[i]][i]
        coeffMatrix[i][i] += 1
        RHS[i][0] = np.sum(transitionMatrix[policy[i]][i]*rewardMatrix[policy[i]][i])
    ans = np.zeros((S, 1))
    valueList = np.matmul(np.linalg.inv(coeffMatrix[considerStates, :][:, considerStates]), RHS[considerStates, :])
    ans[considerStates, :] = list(valueList)
    return np.reshape(ans, (1, S))

prefix = "/home/sainath/Documents/Sem5/CS747/Assignment2/code/data/"

randomPolicyFile = open(prefix + "cricket/rand_pol.txt", 'r')
randomPolicyLines = randomPolicyFile.readlines()

parametersFile = open(prefix + "cricket/sample-p1.txt", 'r')
parameterLines = parametersFile.readlines()

probs = np.zeros((5, 7))

for i in range(1, 6):
    parameterLine = parameterLines[i].strip().split()
    for j in range(1, 8):
        probs[i-1][j-1] = float(parameterLine[j])

stateLines = []
states = []
actions = np.array([0, 1, 2, 4, 6])
actionMap = {"0":0, "1": 1, "2" : 2, "4" : 3, "6" : 4}

randomPolicy = []

for line in randomPolicyLines:
    line = line.strip().split()
    stateLines.append(line[0])
    randomPolicy.append(int(actionMap[line[1]]))

randomPolicy += randomPolicy
randomPolicy += [0, 0]
randomPolicy = np.array(randomPolicy)

stateLinesA = [(i + "0") for i in stateLines]
stateLinesB = [(i + "1") for i in stateLines]
states = stateLinesA + stateLinesB
states += ["W", "L"]

S = len(states)
considerStates = np.arange(0, S-2)

outcomes = np.array([-1, 0, 1, 2, 3, 4, 6])
boutcomes = np.array([-1, 0, 1])
A = len(actions)

mapStrToState = {}
mapStateToStr = {}
for i in range(len(states)):
    mapStrToState[states[i]] = i
    mapStateToStr[i] = states[i]

discount = 1.0
transitions = []

transitionMatrix = np.zeros((A, S, S))
rewardMatrix = np.zeros((A, S, S))

rewardMatrix[:, :, S-2] = 1
rewardMatrix[:, S-2, S-2] = 0
rewardMatrix[:, S-1, S-2] = 0

epsilon = 1e-6

def makeState(b, r):
    stateStr = (str(b)).zfill(2) + (str(r)).zfill(2) + "1"
    return stateStr

x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

qList = np.arange(0, 1, 0.01)
for q in qList:
    transitionMatrix = np.zeros((A, S, S))
    for i in range(S-2):
        balls = int(states[i][0:2])
        runs = int(states[i][2:4])
        player = int(states[i][4])
        if (player == 0):
            for j in range(len(actions)):
                for k in range(len(outcomes)):
                    if (probs[j][k] > 1e-6):
                        if ((outcomes[k] == -1) or ((balls == 1) and (runs > outcomes[k]))):
                            transitionMatrix[j][i][S-1] += probs[j][k] 
                        elif (runs <= outcomes[k]):
                            transitionMatrix[j][i][S-2] += probs[j][k]
                        else :
                            scored = int(outcomes[k])
                            ballsNew = balls - 1
                            runsNew = runs - scored
                            playerNew = player
                            if ((ballsNew%6 != 0 and scored%2 == 1) or (ballsNew%6 == 0 and scored%2 == 0)):
                                playerNew = 1 - player
                            ballsNew = (str(ballsNew)).zfill(2)
                            runsNew = (str(runsNew)).zfill(2)
                            newStateStr = ballsNew + runsNew + str(playerNew)
                            newState = mapStrToState[newStateStr]
                            transitionMatrix[j][i][newState] += probs[j][k]
        else:
            for j in range(len(actions)):
                for k in range(len(boutcomes)):
                    if (boutcomes[k] == -1):
                        transitionMatrix[j][i][S-1] += q
                    elif (((balls == 1) and (runs > boutcomes[k]))):
                        transitionMatrix[j][i][S-1] += (1-q)/2
                    elif (runs <= boutcomes[k]):
                        transitionMatrix[j][i][S-2] += (1-q)/2
                    else :
                        scored = int(boutcomes[k])
                        ballsNew = balls - 1
                        runsNew = runs - scored
                        playerNew = player
                        if ((ballsNew%6 != 0 and scored%2 == 1) or (ballsNew%6 == 0 and scored%2 == 0)):
                            playerNew = 1 - player
                        ballsNew = (str(ballsNew)).zfill(2)
                        runsNew = (str(runsNew)).zfill(2)
                        newStateStr = ballsNew + runsNew + str(playerNew)
                        newState = mapStrToState[newStateStr]
                        transitionMatrix[j][i][newState] += (1-q)/2
    randomPolicyValues = getValue(randomPolicy, transitionMatrix, rewardMatrix, discount, considerStates)
    value = np.zeros((1, S))
    prevValue = np.zeros((1, S))
    compMatrix = transitionMatrix*(rewardMatrix + discount*value)
    sumOverStates = np.sum(compMatrix, axis = 2)
    optPolicy = np.argmax(sumOverStates, axis = 0)
    value = np.max(sumOverStates, axis = 0)
    value = np.reshape(value, (1, S))
    while (abs(np.max(value - prevValue)) > epsilon):
        prevValue = value
        compMatrix = transitionMatrix*(rewardMatrix + discount*value)
        sumOverStates = np.sum(compMatrix, axis = 2)
        optPolicy = np.argmax(sumOverStates, axis = 0)
        value = np.max(sumOverStates, axis = 0)
        value = np.reshape(value, (1, S))
    x1.append(randomPolicyValues[0][0])
    y1.append(value[0][0])

    if (abs(q - 0.25) < 1e-6):
        balls = 10
        for runs in range(20, 0, -1):
            stateStr = makeState(balls, runs)
            index = mapStrToState[stateStr]
            x2.append(randomPolicyValues[0][index])
            y2.append(value[0][index])

        runs = 10
        for balls in range(15, 0, -1):
            stateStr = makeState(balls, runs)
            index = mapStrToState[stateStr]
            x3.append(randomPolicyValues[0][index])
            y3.append(value[0][index])

newqList = np.arange(0, 1, 0.1)
x1 = np.array(x1)
y1 = np.array(y1)
plt.plot(qList, x1)
plt.plot(qList, y1)
plt.xlabel("Value of q")
plt.ylabel("Winning probabilities")
plt.legend(["Random policy", "Optimal policy"])
plt.xticks(newqList)
plt.title("Comparison of winning probabilities for different values of q")
plt.savefig("./Plots/plot1.png")
plt.show()

x2 = np.array(x2)
y2 = np.array(y2)
runs = np.arange(20, 0, -1)
plt.plot(runs, x2)
plt.plot(runs, y2)
plt.xlabel("Number of runs")
plt.ylabel("Winning probabilities")
plt.legend(["Random policy", "Optimal policy"])
plt.title("Winning probabilities for 10 balls")
plt.xticks(runs)
plt.savefig("./Plots/plot2.png")
plt.show()

x3 = np.array(x3)
y3 = np.array(y3)
balls = np.arange(15, 0, -1)
plt.plot(balls, x3)
plt.plot(balls, y3)
plt.xlabel("Number of balls")
plt.ylabel("Winning probabilities")
plt.legend(["Random policy", "Optimal policy"])
plt.title("Winning probabilites for 10 runs")
plt.xticks(balls)
plt.savefig("./Plots/plot3.png")
plt.show()