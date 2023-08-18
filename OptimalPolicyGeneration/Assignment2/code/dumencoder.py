import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--states")
parser.add_argument("--parameters")
parser.add_argument("--q")

args = parser.parse_args()
statesPath = args.states
parametersPath = args.parameters
q = float(args.q)

parametersFile = open(parametersPath, 'r')
parameterLines = parametersFile.readlines()

probs = np.zeros((5, 7))

for i in range(1, 6):
    parameterLine = parameterLines[i].strip().split()
    for j in range(1, 8):
        probs[i-1][j-1] = float(parameterLine[j])

statesFile = open(statesPath, 'r')
stateLines = statesFile.readlines()

stateLines = [i.strip().split()[0] for i in stateLines]
states = [str(i) for i in stateLines]
states += ["W", "L"]
S = len(states)
actions = np.array([0, 1, 2, 4, 6])
outcomes = np.array([-1, 0, 1, 2, 3, 4, 6])

mapStrToState = {}
mapStateToStr = {}
for i in range(len(states)):
    mapStrToState[states[i]] = i
    mapStateToStr[i] = states[i]

discount = 1.0
transitions = []

def makeState(b, r):
    stateStr = (str(b)).zfill(2) + (str(r)).zfill(2)
    return stateStr

for i in range(S-2):
    balls = int(states[i][0:2])
    runs = int(states[i][2:4])
    for j in range(len(actions)):
        for k in range(len(outcomes)):
            if (probs[j][k] > 1e-6):
                if ((outcomes[k] == -1) or ((balls == 1) and (runs > outcomes[k]))):
                    transitions.append((i, j, S-1, 0, probs[j][k]))
                elif (runs <= outcomes[k]):
                    transitions.append((i, j, S-2, 1, probs[j][k]))
                else :
                    scored = int(outcomes[k])
                    ballsNew = balls - 1
                    runsNew = runs - scored
                    if ((ballsNew%6 != 0 and scored%2 == 1) or (ballsNew%6 == 0 and scored%2 == 0)):
                        prob = probs[j][k]
                        for B in range(ballsNew - 1, 0, -1):
                            transitions.append((i, j, S-1, 0, prob*q))
                            if (B%6 == 1 and B > 1):
                                if (runsNew == 1):
                                    transitions.append((i, j, S-2, 1, prob*((1-q)/2)))
                                else :
                                    transitions.append((i, j, mapStrToState[makeState(B-1, runsNew)], 0, prob*((1-q)/2)))
                                    runsNew -= 1
                            else :
                                if (runsNew == 1):
                                    transitions.append((i, j, S-2, 1, prob*((1-q)/2)))
                                else :
                                    transitions.append((i, j, mapStrToState[makeState(B, runsNew-1)], 0, prob*((1-q)/2)))
                            prob = prob*((1-q)/2)
                        transitions.append((i, j, S-1, 0, prob*q))
                        if (runsNew == 1):
                            transitions.append((i, j, S-2, 1, prob*((1-q)/2)))
                        else :
                            transitions.append((i, j, S-1, 0, prob*((1-q)/2)))
                    else :
                        newStateStr = makeState(ballsNew, runsNew)
                        newState = mapStrToState[newStateStr]
                        transitions.append((i, j, newState, 0, probs[j][k]))

for i in range(len(actions)):
    transitions.append((S-1, i, S-1, 0, 1.0))
    transitions.append((S-2, i, S-2, 0, 1.0))

print("numStates", S)
print("numActions", len(actions))
print("end", S-2, S-1)

for i in transitions:
    print("transition", i[0], i[1], i[2], i[3], i[4])

print("mdptype episodic")
print("discount", 1.0)