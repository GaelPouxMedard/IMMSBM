import TP1
import numpy as np
import random
# from scipy import sparse
import sparse

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


# // region Manipulates the datafiles

# Retrieves graph.txt and nounsPost.txt and converts it to histUsr.txt and tweetsUsr.txt
# histUsr and tweetsUsr have the same number of lines and their position match so we
# have on one side the "message" and on the other the "answer"

# Note that "tweets" here refers to the answer to the message, in the same way as a tweet is the "answer" to the
# history feed; the function still works on Reddit, Spotify, ... corpus.
def getData(folder, seuil=50, propPosts=1.):
    g = TP1.Graph().genFromFile("Data/"+folder+"/graph.txt", directed=True)

    mess = {}
    answer = {}
    listInfs = []

    print("Count occurences")
    cntNouns = {}
    totLen=0

    with open("Data/"+folder+"/nounsPost.txt", encoding="utf-8") as f:
        for l in f:
            id, nouns = l.replace("\n", "").split("\t")
            if nouns!=[]:
                for n in nouns.split(" "):
                    try:
                        cntNouns[n] += 1
                    except:
                        cntNouns[n] = 1
            totLen+=1

    
    idToSave=set()

    print("Building contentPosts")
    i=0
    with open("Data/"+folder+"/nounsPost.txt", encoding="utf-8") as f:
        for l in f:
            if i%10000==0:
                print("ContentPosts -", i*100./totLen, "%")

            if random.random() > propPosts:
                continue

            id, nouns = l.replace("\n", "").split("\t")

            for rt in nouns.split(" "):
                if cntNouns[rt] > seuil:
                    rt = rt.lower()
                    listInfs.append(rt)
                    try:
                        answer[id].append(rt)
                    except:
                        answer[id] = [rt]
                    


            if id in g.graphDict:
                for u in g.graphDict[id]:
                    for rt in nouns.split(" "):
                        if cntNouns[rt] > seuil:
                            idToSave.add(id)
                            rt = rt.lower()
                            listInfs.append(rt)
                            try:
                                mess[u].append(rt)
                            except:
                                mess[u] = [rt]

            i += 1
            
            
    print(i, totLen)
   

    # Epurate the corpus so we only keep the pairs message-answer where none of them are empty
    keysMess = list(mess.keys())
    for u in keysMess:
        if u not in answer:
            del mess[u]
        
        elif answer[u]==[] or answer[u]==[""]:
            del answer[u]
            del mess[u]
        
            
    keysAns = list(answer.keys())
    for u in keysAns:
        if u not in mess:
            del answer[u]
        
        elif mess[u]==[] or mess[u]==[""]:
            del answer[u]
            del mess[u]
        
           
    listInfs=set()
    for u in answer:
        for inf in answer[u]:
            listInfs.add(inf)
        for inf in mess[u]:
            listInfs.add(inf)

    

    listInfs = list(sorted(set(listInfs)))

    del g
    return mess, answer, listInfs

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

# Saves the data
def writeToFile_data(folder, histUsr=None, tweetsUsr=None, listInfs=None, alpha_Tr=None, alpha_Te=None, propTrainingSet=None, propPosts=None):
    while True:
        try:
            if histUsr is not None:
                with open("Data/" + folder + "/Inter_tweetsUsr.txt", "a", encoding="utf-8") as f:
                    f.truncate(0)
                with open("Data/" + folder + "/Inter_histUsr.txt", "a", encoding="utf-8") as f:
                    f.truncate(0)
                for u in histUsr:
                    if u not in tweetsUsr:
                        continue
                    if tweetsUsr[u]==[]:
                        continue
                    if histUsr[u]==[]:
                        continue
                        
                        
                    with open("Data/"+ folder + "/Inter_histUsr.txt", "a", encoding="utf-8") as f:
                        f.write(u + "\t")
                        premPass = True
                        for v in histUsr[u]:
                            if not premPass:
                                f.write(" ")
                            f.write(str(v))
                            premPass = False
                        f.write("\n")

                    with open("Data/" + folder + "/Inter_tweetsUsr.txt", "a", encoding="utf-8") as f:
                        if u in tweetsUsr:
                            f.write(u + "\t")
                            premPass = True
                            for v in tweetsUsr[u]:
                                if not premPass:
                                    f.write(" ")
                                f.write(str(v))
                                premPass = False
                            f.write("\n")
                        else:
                            f.write(u + "\t-1\n")



            if listInfs is not None:
                with open("Data/"+ folder + "/Inter_listInfs.txt", "a", encoding="utf-8") as f:
                    f.truncate(0)
                    i = 0
                    for u in listInfs:
                        f.write(str(i) + "\t" + u + "\n")
                        i += 1

            if propTrainingSet is not None:
                f = open("Data/"+ folder + "/Inter_propTrainingSet.txt", "w+", encoding="utf-8")
                f.write(str(propTrainingSet))
                f.close()

            if propPosts is not None:
                f = open("Data/"+ folder + "/Inter_propPosts.txt", "w+", encoding="utf-8")
                f.write(str(propPosts))
                f.close()

            if alpha_Tr is not None and alpha_Te is not None:
                writeMatrix(alpha_Tr, "Data/"+ folder + "/Inter_alpha_Tr.txt")
                writeMatrix(alpha_Te, "Data/"+ folder + "/Inter_alpha_Te.txt")

            break


        except Exception as e:
            print("Retrying to write file -",e)
            pass


# // endregion

# From alpha and beta, saves each triplet input-input-output in the form (i,j,k,alpha_{i,j,k},beta_{i,j,k})
# where i and j are inputs and k is the resulting ouput
# Saving this is useful for building the ROC curve
def writePairs(alpha, beta, propTr, folder, TrTe):
    pairs=[]
    setInd = set()
    nonZ=alpha.nonzero()
    for (i, j, k) in zip(nonZ[0], nonZ[1], nonZ[2]):
        setInd.add((i,j,k))
    del nonZ
    nonZ=beta.nonzero()
    for (i, j, k) in zip(nonZ[0], nonZ[1], nonZ[2]):
        if alpha[i,j,k]!=0:
            setInd.add((i,j,k))
    for (i,j,k) in setInd:
        pairs.append((i, j, k))

    pairs = list(sorted(pairs))

    pairs = pairs[:int(len(pairs)*1)]

    with open(folder + "/Inter_Pairs_"+TrTe+".txt", "a") as f:
        f.truncate(0)

    iterPerc, iter=0, 0
    totL = len(pairs)
    with open(folder + "/Inter_Pairs_" + TrTe + ".txt", "a") as f:
        for (i, j, k) in pairs:
            if iterPerc % 100000 == 0:
                print(iterPerc * 100. / totL)
            f.write(str((i, j, k, alpha[i, j, k], beta[i, j, k])) + "\n")

            iter += 1
            iterPerc += 1

# Counts entities in a sequence and saves it into a vector.
# The "unit" parameter decides whether we count the number of times an entity appears or just assess its presency
def seqWordsToVec(seq, infToInt, unit=True):
    vec = np.zeros(len(infToInt), dtype=int)
    for a in seq:
        vec[infToInt[a]] += 1
    if not unit:
        return vec
    else:
        return np.array((vec > 0).astype(int))

# Computes alpha of dim=3, whose form is alpha_{input1, input2, output}, from histUsr and tweetsUsr
def getAlpha(listInfs, propTrainingSet, folder, TrTe, indPosts):
    infToInt = {}
    ind = 0
    for i in listInfs:
        infToInt[i] = ind
        ind += 1

    I = len(listInfs)

    dataAlphaComp = {}
    dataBetaComp = {}

    random.seed(seed)
    listTest = random.sample(indPosts, int((1.-propTrainingSet)*len(indPosts)))
    with open(folder + "/Inter_testMessages.txt", "w") as f:
        for u in listTest:
            f.write(str(u)+"\n")
    nbPosts = len(indPosts)

    iter = 0
    with open(folder + "/Inter_histUsr.txt", "r", encoding="utf-8") as f_hi:
        with open(folder + "/Inter_tweetsUsr.txt", "r", encoding="utf-8") as f_tw:
            for lHist in f_hi:
                lHist=lHist.replace("\n", "")
                u, nouns = lHist.split("\t")
                nouns=nouns.split(" ")

                uTw, nounsTw = f_tw.readline().replace("\n","").split("\t")
                nounsTw=nounsTw.split(" ")

                iter += 1
                if iter % 100 == 0 or False:
                    print("a -", iter * 100. / nbPosts, "%")

                if nounsTw == ["-1"]:
                    continue
                if (uTw not in listTest and TrTe == "Te") or (uTw in listTest and TrTe == "Tr"):
                    continue

                vecWdsHist = seqWordsToVec(nouns, infToInt, unit=False)
                vecWdsTwts = seqWordsToVec(nounsTw, infToInt, unit=True)
                indHist, indTwts = vecWdsHist.nonzero()[0], vecWdsTwts.nonzero()[0]
                #indTwtsZero = np.setdiff1d(range(I), indTwts)

                if np.sum(vecWdsHist)>15:  # On considère ce cas comme peu informatif (225 paires pour une réponse ?)
                    continue

                for i in indHist:
                    nbi = vecWdsHist[i]
                    for j in indHist:
                        nbj = vecWdsHist[j]
                        for k in indTwts:
                            try:
                                dataAlphaComp[i][j][k] += nbi * nbj
                            except:
                                try:
                                    dataAlphaComp[i][j][k] = nbi * nbj
                                except:
                                    try:
                                        dataAlphaComp[i][j] = {}
                                        dataAlphaComp[i][j][k] = nbi * nbj
                                    except:
                                        try:
                                            dataAlphaComp[i] = {}
                                            dataAlphaComp[i][j] = {}
                                            dataAlphaComp[i][j][k] = nbi * nbj
                                        except:
                                            pass


    iter = 0
    with open(folder + "/Inter_histUsr.txt", "r", encoding="utf-8") as f_hi:
        with open(folder + "/Inter_tweetsUsr.txt", "r", encoding="utf-8") as f_tw:
            for lHist in f_hi:
                lHist=lHist.replace("\n", "")
                u, nouns = lHist.split("\t")
                nouns=nouns.split(" ")

                uTw, nounsTw = f_tw.readline().replace("\n","").split("\t")
                nounsTw=nounsTw.split(" ")

                if iter % 100 == 0 or False:
                    print("b -", iter * 100. / nbPosts, "%")
                iter += 1

                if nounsTw == ["-1"]:
                    continue
                if (uTw not in listTest and TrTe == "Te") or (uTw in listTest and TrTe == "Tr"):
                    continue

                vecWdsHist = seqWordsToVec(nouns, infToInt, unit=False)
                vecWdsTwts = seqWordsToVec(nounsTw, infToInt, unit=True)
                indHist, indTwts = vecWdsHist.nonzero()[0], vecWdsTwts.nonzero()[0]
                # indTwtsZero = np.where(vecWdsTwts == 0)[0]

                if np.sum(vecWdsHist)>15:  # On considère ce cas comme peu informatif (225 paires pour une réponse ?)
                    continue

                for i in dataAlphaComp:
                    if i not in indHist:
                        continue
                    nbi = vecWdsHist[i]
                    for j in dataAlphaComp[i]:
                        if j not in indHist:
                            continue
                        nbj = vecWdsHist[j]

                        for k in dataAlphaComp[i][j]:
                            if k in indTwts:
                                continue
                            try:
                                dataBetaComp[i][j][k] += nbi * nbj
                            except:
                                try:
                                    dataBetaComp[i][j][k] = nbi * nbj
                                except:
                                    try:
                                        dataBetaComp[i][j] = {}
                                        dataBetaComp[i][j][k] = nbi * nbj
                                    except:
                                        try:
                                            dataBetaComp[i] = {}
                                            dataBetaComp[i][j] = {}
                                            dataBetaComp[i][j][k] = nbi * nbj
                                        except:
                                            print("PROBLEM CONSTRUCT BETA")


    coords, dataA, dataB = [[], [], []], [], []

    for i in dataAlphaComp:
        for j in dataAlphaComp[i]:
            for k in dataAlphaComp[i][j]:
                coords[0].append(i)
                coords[1].append(j)
                coords[2].append(k)
                dataA.append(dataAlphaComp[i][j][k])

                try:
                    dataB.append(dataBetaComp[i][j][k])
                except:
                    dataB.append(0)

    del dataAlphaComp
    del dataBetaComp

    alpha = sparse.COO(coords, dataA, shape=(I, I, I), )
    del dataA
    beta = sparse.COO(coords, dataB, shape=(I, I, I), )
    del coords
    del dataB

    # beta = beta * (alpha != 0).astype(int)

    print("Writing pairs")
    writePairs(alpha, beta, propTrainingSet, folder, TrTe)
    del beta

    return alpha


def getObservations(folder, propTrainingSet, retreatEverything=False, seuil=None):
    propPosts = 1.

    if folder=="SymptomeDisease":  # As stated in the article, we do not consider every pubmed publication but
                                   # only 60% of them
        propPosts = 0.6

    print("Get data")
    mess, answer, listInfs = getData(folder, seuil, propPosts)

    indPosts = list(answer.keys())

    print("Saving data 1")
    writeToFile_data(folder, mess, answer, listInfs)
    del mess
    del answer
    print("Number infs :", len(listInfs))


    print("Compute alpha")
    alpha_Tr = getAlpha(listInfs, propTrainingSet, folder="Data/"+folder+"/", TrTe="Tr", indPosts=indPosts)  # Training set
    alpha_Te = getAlpha(listInfs, propTrainingSet, folder="Data/"+folder+"/", TrTe="Te", indPosts=indPosts)  # Test set
        
        

    print("Saving data 2")
    writeToFile_data(folder, None, None, None, alpha_Tr, alpha_Te, propTrainingSet, propPosts)

    return alpha_Tr, alpha_Te









