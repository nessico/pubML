import random
import numpy as np
from sklearn.preprocessing import normalize

# SJF AND FCFS calculation is contributed from ChitraNayal and Shubham Singh

# Shortest Job First Calculation


def SJFWaitingTime(processes, n, wt):
    rt = [0] * n

    # Copy the burst time into rt[]
    for i in range(n):
        rt[i] = processes[i][1]
    complete = 0
    t = 0
    minm = 999999999
    short = 0
    check = False

    # Process until all processes gets
    # completed
    while (complete != n):

        # Find process with minimum remaining
        # time among the processes that
        # arrives till the current time`
        for j in range(n):
            if ((processes[j][2] <= t) and
                    (rt[j] < minm) and rt[j] > 0):
                minm = rt[j]
                short = j
                check = True
        if (check == False):
            t += 1
            continue

        # Reduce remaining time by one
        rt[short] -= 1

        # Update minimum
        minm = rt[short]
        if (minm == 0):
            minm = 999999999

        # If a process gets completely
        # executed
        if (rt[short] == 0):

            # Increment complete
            complete += 1
            check = False

            # Find finish time of current
            # process
            fint = t + 1

            # Calculate waiting time
            wt[short] = (fint - proc[short][1] -
                         proc[short][2])

            if (wt[short] < 0):
                wt[short] = 0

        # Increment time
        t += 1


def SJFTurnAroundTime(processes, n, wt, tat):

    for i in range(n):
        tat[i] = processes[i][1] + wt[i]


def SJFaverageTime(processes, n):
    wt = [0] * n
    tat = [0] * n

    SJFWaitingTime(processes, n, wt)

    SJFTurnAroundTime(processes, n, wt, tat)
    print("SJF Calculation:")

    print("Processes    Burst Time     Waiting",
          "Time     Turn-Around Time")
    total_wt = 0
    total_tat = 0
    for i in range(n):

        total_wt = total_wt + wt[i]
        total_tat = total_tat + tat[i]
        print(" ", processes[i][0], "\t\t",
              processes[i][1], "\t\t",
              wt[i], "\t\t", tat[i])

    print("Average waiting time = %.5f " % (total_wt / n))
    print("Average turn around time = ", total_tat / n, "\n")
    waitFinal = total_wt / n
    turnFinal = total_tat / n
    SJFresult.append(waitFinal)
    SJFresult.append(turnFinal)


# First Come First Serve Calculation

def FCFSWaitingTime(processes, n,
                    bt, wt):

    wt[0] = 0

    for i in range(1, n):
        wt[i] = bt[i - 1] + wt[i - 1]


def FCFSTurnAroundTime(processes, n,
                       bt, wt, tat):

    for i in range(n):
        tat[i] = bt[i] + wt[i]


def FCFSaverageTime(processes, n, bt):

    wt = [0] * n
    tat = [0] * n
    total_wt = 0
    total_tat = 0

    FCFSWaitingTime(processes, n, bt, wt)
    FCFSTurnAroundTime(processes, n,
                       bt, wt, tat)

    print("FCFS Calculation:")
    print("Processes Burst time " +
          " Waiting time " +
          " Turn around time")
    for i in range(n):

        total_wt = total_wt + wt[i]
        total_tat = total_tat + tat[i]
        print(" " + str(i + 1) + "\t\t" +
              str(bt[i]) + "\t " +
              str(wt[i]) + "\t\t " +
              str(tat[i]))

    print("Average waiting time = " +
          str(total_wt / n))
    print("Average turn around time = " +
          str(total_tat / n) + "\n")
    waitf = total_wt / n
    turnf = total_tat / n
    FCFSresult.append(waitf)
    FCFSresult.append(turnf)

# Neural Network


def nonlin(x, deriv=False):
    if(deriv):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def NNetwork(SJFresult, FCFSresult, normalizeOutput):
    normInput1 = SJFresult.copy()

    for i in FCFSresult:
        normInput1.append(i)
    normInput2 = (normInput1[0] + SJFresult[1] +
                  normInput1[2] + normInput1[3]) / 4
    for x in range(0, 4):
        if normInput1[x] >= normInput2:
            normInput1[x] = 1
        else:
            normInput1[x] = 0

    finput = np.array([normInput1])
    originalInput = np.array(normInput1)

    for x in range(0, 3):
        temp1 = normInput1.copy()
        random.shuffle(temp1)

        temp1 = np.array([temp1])
        finput = np.concatenate((finput, temp1), axis=0)

    syn0 = 2 * np.random.random((4, 1)) - 1

    foutput = np.array([normalizeOutput]).T
    print("Normalized Input Layer:")
    print(finput)
    print("Normalized Output Layer:")
    print(foutput)
    print("Training Data...")

    for iter in range(1000):
        inputlayer = finput
        hiddenlayer = nonlin(np.dot(inputlayer, syn0))
        hiddenlayer_error = foutput - hiddenlayer
        hiddenlayer_delta = hiddenlayer_error * nonlin(hiddenlayer, True)
        syn0 += np.dot(inputlayer.T, hiddenlayer_delta)

    print("Output Layer After Training:")
    print(hiddenlayer)
    f1 = np.asarray(originalInput, dtype='float64')
    f2 = np.asarray(hiddenlayer, dtype='float64')
    prediction = 1 / (1 + np.exp(-(np.dot(f1, f2))))
    print("Prediction Value: ")
    print(prediction)

    val = prediction.item(0)
    if val >= .8:
        print("The Algorithm predicts that SJF will have a HIGH probability to \ncompute the faster average waiting time and average turnaround time, because the value is very close to 1.")
    elif val > .6:
        print("The Algorithm predicts that SJF will have a medium-high probability to \ncompute the faster average waiting time and average turnaround time, because the value is reasonably greater than .5.")
    elif .5 <= val <= .6:
        print("The Algorithm predicts that both algorithms will have similar efficiency, because the value is very close to .5.")
    elif .2 < val <= .4:
        print("The Algorithm predicts that FCFS will have a medium-high probability to \ncompute the faster average waiting time and average turnaround time, because the value is reasonably lower than .5.")
    elif val <= .2:
        print("The Algorithm predicts that FCFS will have a HIGH probability to \ncompute the faster average waiting time and average turnaround time, because the value is very cloes to 0.")


# Main
if __name__ == "__main__":
    SJFresult = []
    FCFSresult = []
    # Assigning randomized Burst times
    b1 = random.randint(1, 100)
    b2 = random.randint(1, 100)
    b3 = random.randint(1, 100)
    b4 = random.randint(1, 100)
    # Process ID, Burst Time, Order
    proc = [[1, b1, 1], [2, b2, 1],
            [3, b3, 2], [4, b4, 3]]

    nsjf = 4
    SJFaverageTime(proc, nsjf)

    # FCFS
    # Process ID
    processes = [1, 2, 3, 4]
    nfcfs = len(processes)

    # Burst Time
    burst_time = [b1, b2, b3, b4]
    FCFSaverageTime(processes, nfcfs, burst_time)

    normalizeOutput = [b1, b2, b3, b4]
    norm1 = (b1 + b2 + b3 + b4) / 4
    for x in range(0, 4):
        if normalizeOutput[x] >= norm1:
            normalizeOutput[x] = 1
        else:
            normalizeOutput[x] = 0

    NNetwork(SJFresult, FCFSresult, normalizeOutput)
