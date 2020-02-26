import numpy as np
import random
# from scipy import sparse
import sparse
import datetime

'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
pause()

'''

seed = 1111


# // region Manipulates the data files

# Generic function to save matrices of dim=2 or 3
def writeMatrix(arr, filename):
    try:
        sparse.save_npz(filename.replace(".txt", ""), arr)
    except:

        with open(filename, 'a') as outfile:
            outfile.truncate(0)
            outfile.write('# Array shape: {0}\n'.format(arr.shape))
            for slice_2d in arr:
                np.savetxt(outfile, slice_2d)
                outfile.write("# New slice\n")

    # np.savetxt(filename, arr)

# Generic function to read matrices of dim=2 or 3
def readMatrix(filename):
    try:
        return sparse.load_npz(filename.replace(".txt", ".npz"))
    except:
        with open(filename, 'r') as outfile:
            dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
            for i in range(len(dims)):
                dims[i] = int(dims[i])

        new_data = np.loadtxt(filename).reshape(dims)
        return new_data

    # return sparse.csr_matrix(new_data)

# Saves the model's parameters theta, p, corresponding likelihood and held-out likelihood
def writeToFile_params(folder, theta, p, maxL, HOL, T, selfInter=False, run=-1):
    while True:
        try:
            I=len(theta)
            if selfInter:
                s="Self"
            else:
                s=""
            folderParams = "Output/" + folder + "/"
            pass
            writeMatrix(theta, folderParams + "/T=%.0f_%.0f_" % (T, run)+s+"Inter_theta.txt" )

            writeMatrix(p, folderParams + "/T=%.0f_%.0f_" % (T, run)+s+"Inter_p.txt")

            f = open(folderParams + "/T=%.0f_%.0f_" % (T, run)+s+"Inter_L.txt", "w")
            f.write(str(maxL) + "\n")
            f.close()

            f = open(folderParams + "/T=%.0f_%.0f_" % (T, run)+s+"Inter_HOL.txt", "w")
            f.write(str(HOL) + "\n")
            f.close()

            break

        except Exception as e:
            print("Retrying to write file -", e)

# Get alpha training and test from file
def recoverData(folder):
    folderData = "Data/"+folder+"/"
    alpha_Tr, alpha_Te = readMatrix(folderData + "/Inter_alpha_Tr.txt"), readMatrix(folderData + "/Inter_alpha_Te.txt")


    return alpha_Tr, alpha_Te


# // endregion


# // region Fit tools

# Computes the likelihood of the model
def likelihood(theta, p, alpha):
    L = 0.
    I=len(alpha)

    '''
    I = len(alpha)
    for i in range(I):
        for j in range(I):
            for inf in range(I):
                temp = 0
                for k in range(T):
                    for l in range(T):
                        temp+=theta[j,l]*theta[i,k]*p[k,l,inf]

                L+=alpha[i,j,inf]*np.log(temp+1e-10)


    I = len(alpha)
    probs = theta.dot((theta.dot(p)))
    for i in range(I):
        for j in range(I):
            for inf in range(I):
                tmp=0.
                for t in range(T):
                    for s in range(T):
                        tmp+=theta[i,t]*theta[j,s]*p[t,s,inf]
                print(probs[i, j, inf] - tmp)

    '''


    coords = alpha.nonzero()
    vals=[]
    for (i,j,k) in zip(coords[0], coords[1], coords[2]):
        vals.append(theta[j].dot(theta[i].dot(p[:, :, k])))
    probs = sparse.COO(coords, vals, shape=(I,I,I))

    L = (alpha * (np.log(1e-10 + probs))).sum()

    # L = np.sum(alpha * (np.log(1e-10 + theta.dot((theta.dot(p))))))

    return L

# EM steps for theta
def maximization_Theta(alpha, I, T, thetaPrev, p):
    '''
    theta2 = np.zeros((I, T))
    for m in range(I):
        nonZG, nonZD = alpha[m, :, :].nonzero(), alpha[:, m, :].nonzero()
        for n in range(T):
            tmp=0.
            for s in range(T):
                for (i, inf) in zip(nonZG[0], nonZG[1]):
                    tmp+=alpha[m, i, inf] * omega[m, i, s, n, inf]
                for (i, inf) in zip(nonZD[0], nonZD[1]):
                    tmp+=alpha[i, m, inf] * omega[i, m, n, s, inf]

            theta2[m,n] = tmp / Cm[m]
    '''

    '''  Memory consuming
    divix = (thetaPrev.dot(thetaPrev.dot(p))) + 1e-10  # mrx
    divix = np.swapaxes(divix, 0, 1)  # rmx  # Parce que alpha c'est dans l'ordre rmx

    terme1 = np.swapaxes(alpha/divix, 0, 1)  # mrx
    terme2 = np.swapaxes(thetaPrev.dot(p), 1, 2)  # rxk
    theta = np.tensordot(terme1, terme2, axes=2)  # mk


    terme1 = np.swapaxes(terme1, 0, 1)  # rmx
    terme2 = np.swapaxes(thetaPrev.dot(np.swapaxes(p, 0, 1)), 1, 2)  # mxl
    theta += np.tensordot(terme1, terme2, axes=2)  # rl

    theta = theta / Cm[:, None]
    theta *= thetaPrev
    '''

    # Combinaisons : rl, mk, klx  ;  alpha(rmx)!=alpha(mrx) car on considere ici alpha_Tr

    coords = alpha.nonzero()
    vals=[]
    for (r,m,k) in zip(coords[0], coords[1], coords[2]):
        vals.append(thetaPrev[r].dot(thetaPrev[m].dot(p[:, :, k])))  # rmx
    divix = sparse.COO(coords, np.array(vals), shape=(I,I,I))+1e-10

    Cm = (alpha.sum(axis=0).sum(axis=1) + alpha.sum(axis=1).sum(axis=1)).todense()+1e-10

    terme1 = alpha / divix  # rmx
    terme2 = np.swapaxes(thetaPrev.dot(np.swapaxes(p, 0, 1)), 1, 2)  # mxl
    theta = sparse.tensordot(terme1, terme2, axes=2)  # rl

    terme1 = terme1.transpose(axes=(1, 0, 2))  # mrx
    terme2 = np.swapaxes(thetaPrev.dot(p), 1, 2)  # rxk
    theta += sparse.tensordot(terme1, terme2, axes=2)  # mk

    theta = theta / Cm[:, None]
    theta *= thetaPrev

    return theta

# EM steps for p
def maximization_p(alpha, I, T, theta, pPrev):
    '''
    nonZ = alpha.nonzero()

    p2 = np.zeros((T, T, I))
    for m in range(T):
        for n in range(T):
            div = 0.
            for (i, j, inf) in zip(nonZ[0], nonZ[1], nonZ[2]):
                div+=alpha[i,j,inf]*omega[i,j,m,n,inf]


            for (i, j, inf) in zip(nonZ[0], nonZ[1], nonZ[2]):
                p2[m,n,inf] += alpha[i,j,inf]*omega[i,j,m,n,inf]


            p2[m,n] = p2[m,n, :]/div
    '''

    ''' Memory consuming
    divrm = (theta.dot(theta.dot(np.swapaxes(pPrev, 0, 1)))) + 1e-10  # rmx

    terme1 = np.swapaxes(alpha/divrm, 0, 2)  # xmr
    p = np.tensordot(terme1, theta, axes=1)  # xml
    p = np.swapaxes(p, 1, 2)  # xlm
    p = np.tensordot(p, theta, axes=1)  # xlk
    p = np.swapaxes(p, 0, 2)  # klx

    grandDiv = np.sum(p * pPrev, axis=2)[:, :, None] + 1e-10
    p = p * pPrev / grandDiv
    '''


    coords = alpha.nonzero()
    vals=[]
    for (r,m,k) in zip(coords[0], coords[1], coords[2]):
        vals.append(theta[r].dot(theta[m].dot(pPrev[:, :, k])))  # rmx
    divrm = sparse.COO(coords, np.array(vals), shape=(I,I,I))+1e-10

    terme1 = (alpha/divrm).transpose((2, 1, 0))  # xmr
    p = terme1.dot(theta)  # xml
    p = p.transpose((0, 2, 1))  # xlm
    p = p.dot(theta)  # xlk
    p = p.transpose((2, 1, 0))  # klx

    grandDiv = np.sum(p * pPrev, axis=2)[:, :, None] + 1e-10
    p = p * pPrev / grandDiv

    return p

# Random initialisation of p, theta
def initVars(I, T):
    theta, p = np.random.rand(I, T), np.random.random((T, T, I))

    for k in range(I):
        p[:,:,k]=(p[:,:,k]+p[:,:,k].T)/2

    p = p / np.sum(p, axis=2)[:, :, None]
    theta = theta / np.sum(theta, axis=1)[:, None]

    return theta, p

# Main loop of the EM algorithm, for 1 run
def EMLoop(alpha, T, I, maxCnt, prec, alpha_Te, folder, selfInter, run):
    theta, p = initVars(I, T)
    maxTheta, maxP = initVars(I, T)

    prevL, L, maxL = -1e10, 0., 0.
    cnt = 0

    i = 0
    iPrev=0
    while i < 1000:  # 1000 iterations top ; prevent infinite loops but never reached in practice
        print(i)

        if i%10==0:  # Compute the likelihood and possibly save the results every 10 iterations
            L = likelihood(theta, p, alpha)
            print("L =", L)

            if abs((L - prevL) / L) < prec:
                cnt += i-iPrev
                if cnt > maxCnt:
                    break
            else:
                cnt = 0

            iPrev=i

            if L > prevL:
                maxTheta, maxP = theta, p
                maxL = L
                HOL = likelihood(theta, p, alpha_Te)
                writeToFile_params(folder, maxTheta, maxP, maxL, HOL, T, selfInter, run)
                print("Saved")

            prevL = L
        thetaNew = maximization_Theta(alpha, I, T, theta, p)
        pNew = maximization_p(alpha, I, T, theta, p)
        p = pNew
        theta = thetaNew

        i += 1

    return maxTheta, maxP, maxL


# // endregion


def runFit(folder, T, prec, maxCnt, saveToFile, propTrainingSet, treatData, nbRuns, retreatEverything, seuil, selfInter):
    np.random.seed(seed)
    random.seed(seed)

    if treatData:
        import MMSBM_buildObservations
        alpha_Tr, alpha_Te = MMSBM_buildObservations.getObservations(folder, propTrainingSet, retreatEverything, seuil)

    else:
        alpha_Tr, alpha_Te = recoverData(folder)

    I = len(alpha_Tr)

    print(I, "infs")

    alpha_Tr=alpha_Tr[:I, :I, :I]
    alpha_Te=alpha_Te[:I, :I, :I]


    maxL = -1e100
    for i in range(nbRuns):
        print("RUN", i)
        theta, p, L = EMLoop(alpha_Tr, T, I, maxCnt, prec, alpha_Te, folder, selfInter, i)
        HOL = likelihood(theta, p, alpha_Te)
        if L > maxL:
            maxL = L
            writeToFile_params(folder + "/Final/", theta, p, L, HOL, T, selfInter, -1)
            print("######saved####### MAX L =", L)
        print("=============================== END EM ==========================")


# Run the algorithm with command line parameters or not
# TreatData = do you want to redo the entire corpus from raw data
# RetreatEverything = do you want to compute alpha again
# folder = has to do with the project structure
import sys
try:
    folder=sys.argv[1]
    treatData=int(sys.argv[2])
    retreatEverything=int(sys.argv[3])
except Exception as e:
    treatData = True  # BuildObservations.py
    retreatEverything = False  # TreatData.py
    folder = "Reddit"
    
selfInter=False


prec = 1e-4  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
maxCount = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
saveToFile = True
propTrainingSet = 0.9
nbRuns = 100

print(folder)
print("Self inter =", selfInter)

seuil=0  # If retreatEverything=True : choose the threshold for the number of apparitions of an entity.
# If an entity appears less than "seuil" times, it's not included in the dataset
if folder=="Spotify":
    seuil=20
elif folder=="Reddit":
    seuil=200
elif folder=="SymptomeDisease":
    seuil=0
elif folder=="Twitter":
    seuil=50
else:
    print("WRONG INPUT")

# Number of clusters considered. The algorithm does nbRuns runs for each of those numbers.
listT = [3, 5, 10, 15, 20, 25, 30, 40, 50]
for T in listT:
    print("================ %.0f ================" %T)
    runFit(folder, T, prec, maxCount, saveToFile, propTrainingSet, treatData, nbRuns, retreatEverything, seuil, selfInter)
    treatData=False

