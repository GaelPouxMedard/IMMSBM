import TP1
import json

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import string
punctuations = list(string.punctuation)
import inflect
InflectEngine = inflect.engine()

import spacy
spacy_nlp = spacy.load('en_core_web_sm')


'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")

'''

def preprocess_nltk(txt):
    for punc in punctuations:
        txt=txt.replace(punc, "")
    txt = nltk.word_tokenize(txt)
    txt = nltk.pos_tag(txt)
    return txt

def preprocess_spacy(txt):
    for punc in punctuations:
        txt=txt.replace(punc, "")
    txt = spacy_nlp(txt)
    
    return txt

def saveData(folder, g, nounsPost):
    while True:
        try:
            listInfs=set()
            with open(folder + "nounsPost.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in nounsPost:
                    f.write(str(u) + "\t")
                    premPass = True
                    for v in nounsPost[u]:
                        if not premPass:
                            f.write(" ")
                        f.write(str(v))
                        listInfs.add(v)
                        premPass = False
                    f.write("\n")

            with open(folder+"graph.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in g.graphDict:
                    for v in g.graphDict[u]:
                        f.write(str(v)+"\t"+str(u)+"\n")


                break

        except Exception as e:
            print("Retrying to write file :", e)
            pass


def preprocess_word(m):
    m=m.lower()
    #m = delete repetitions
    try:
        sing=InflectEngine.singular_noun(m)
        if sing is not False:
            m=sing
    except:
        pass

    return m


def retreatZST(folder, listFiles):
    if listFiles is None:
        from os import listdir
        listFiles=listdir("Data/"+folder+"/Raw/")
        temp=[]
        for n in listFiles:
            if "zst" in n:
                temp.append(n)
        listFiles=temp

    #fSubRedNews = open("Data/"+folder+"/Raw/SubRedNews0419.txt", "a")
    #fSubRedNews.truncate(0)
    g={}
    contentPost={}
    import zstandard as zstd
    for file in listFiles:
        with open("Data/"+folder+"/Raw/"+file, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            s = 179369162500  # Taille estimee du fichier decompresse
            with dctx.stream_reader(fh) as reader:
                previous_line=""
                iterChunk = 0
                while True:
                    chunk = reader.read(2**27)
                    iterChunk+=1
                    
                    if not chunk:
                        break

                    string_data = chunk.decode('utf-8')
                    lines = string_data.split("\n")
                    print(iterChunk * 2 ** 27 * 100. / s, "% -", len(lines), " lines")
                    iter=0
                    for iter, line in enumerate(lines[:-1]):
                        if iter == 0:
                            line = previous_line + line
                        if iter > 3000:
                            pass
                            # break

                        # print(line)
                        # print()
                        l = json.loads(line)

                        # print(l)
                        if l["subreddit"]!="news":
                            continue

                        txtPar = l["parent_id"][3:]
                        txtSon = l["id"]

                        try:
                            g[txtPar].append(txtSon)
                        except:
                            g[txtPar] = [txtSon]

                        taggedSent = preprocess_spacy(l["body"])
                        words = []
                        for token in taggedSent:
                            word, func = token.text, token.tag_
                            word = preprocess_word(word)
                            if (func.startswith('NN') or func.startswith('JJ')) and len(word) > 1 and len(word) < 20:
                                words.append(word)

                        contentPost[txtSon] = words

                    previous_line = lines[-1]

    return g, contentPost


def retreatUncompressed(folder, listFiles=None):
    if listFiles is None:
        from os import listdir
        listFiles=listdir("Data/"+folder+"/Raw/")
        temp=[]
        for n in listFiles:
            if "zst" not in n:
                temp.append(n)
        listFiles=temp

        print(listFiles)

    g={}
    contentPost={}
    for file in listFiles:
        lenFile = 0
        for _ in open("Data/"+folder+"/Raw/"+file).readlines(): lenFile += 1
        with open("Data/"+folder+"/Raw/"+file) as f:
            iter=0

            for line in f:
                iter += 1
                if iter>3000:
                    pass
                    #break

                if iter%10000==0:
                    print(iter*100/lenFile, r"%")

                l = json.loads(line)
                #print(l)
                # ['author', 'author_flair_css_class', 'author_flair_text', 'body', 'can_gild', 'controversiality', 'created_utc', 'distinguished', 'edited', 'gilded', 'id', 'is_submitter', 'link_id', 'parent_id', 'permalink', 'retrieved_on', 'score', 'stickied', 'subreddit', 'subreddit_id', 'subreddit_type']

                if l["subreddit"]!="news":
                    continue

                #print(l)

                txtPar = l["parent_id"][3:]
                txtSon = l["id"]

                try:
                    g[txtPar].append(txtSon)
                except:
                    g[txtPar]=[txtSon]


                taggedSent = preprocess_spacy(l["body"])
                words = []
                for token in taggedSent:
                    word, func = token.text, token.tag_
                    word = preprocess_word(word)
                    if func.startswith('NN') and len(word)>1 and len(word)<30:
                        words.append(word)

                contentPost[txtSon] = words
    return g, contentPost


def sliceContentPosts(contentPosts, g, K=1):
    iter=0
    perc=0
    contentPosts2, g2 = {}, {}
    lgTot = len(contentPosts)
    toDel=[]
    toAppend=[]
    for u in contentPosts:
        if perc%10000==0:
            print(perc*100/lgTot, "%")
        perc+=1
        if u not in g:
            continue
        if len(contentPosts[u])<2*K+1:
            toDel.append(u)
            continue
        toAppend.append(u)
        for v in g[u]:
            if v not in contentPosts:
                continue
            if contentPosts[u]==[""] or contentPosts[v]==[""]:
                continue
            for n in range(len(contentPosts[u])):
                packWds=[]
                nbWds=len(contentPosts[u])
                indDeb, indFin=n-K, n+K+1
                indFin=n+1
                if indDeb<0:
                    indDeb=0
                if indFin>nbWds:
                    indFin=nbWds
                for m in range(indDeb, indFin):
                    packWds.append(contentPosts[u][m])

                contentPosts2[iter]=packWds


                contentPosts2[v+str(iter)]=contentPosts[v]
                try:
                    g2[iter].append(v+str(iter))
                except:
                    g2[iter]=[v+str(iter)]

                iter+=1

    iter=0
    #for u in toAppend:
    #    contentPosts2[u]=contentPosts[u]
        

    return g2, contentPosts2


def dataFromFiles():
    folder = "Data/Reddit/nounsPost_noWindow.txt"
    contentPosts={}
    with open(folder, encoding="utf-8") as f:
        for l in f:
            id, mots = l.replace("\n", "").split("\t")
            mots = mots.split(" ")
            contentPosts[id]=mots

    with open(folder.replace("nounsPost", "graph")) as f:  # Sauve a l'envers par rapport au programme qui le traite
        g2 = {}
        for l in f:
            l=l.replace("\n", "")
            v, u = l.split("\t")
            try:
                g2[u].append(v)
            except:
                g2[u]=[v]

    return contentPosts, g2

def epurateCorpus(g, contentPosts, n):
    listKeys = list(contentPosts.keys())
    for u in listKeys:
        if len(contentPosts[u])>n:
            del contentPosts[u]
            
    return contentPosts
        
    
def selectKeyWords(g, contentPost):
    listKeys = list(contentPost.keys())
    KW = ["paris", "cathedral", "notre", "fire", "church", "prayers", "france", "macron", "rebuild", "roof"]

    for u in listKeys:
        pres=False
        for w in contentPost[u]:
            for kw in KW:
                if kw in w:
                    pres=True
                    break
        '''
        if u in g:
            for v in g[u]:
                if v in contentPost:
                    for w in contentPost[v]:
                        for kw in KW:
                            if kw in w:
                                pres=True
                                break
                        if pres: break
                if pres: break
        '''

        if not pres:
            del contentPost[u]

    return g, contentPost




def retreat(folder, listFiles=None):
    #g, contentPost = retreatUncompressed(folder, listFiles)
    #g, contentPost = retreatZST(folder, listFiles)

    contentPost, g = dataFromFiles()
    contentPost = epurateCorpus(g, contentPost, 15)
    
    #g, contentPost = sliceContentPosts(contentPost, g, 1)
    
    #g, contentPost = selectKeyWords(g, contentPost)

        
        

    g = TP1.Graph(g)


    i = 0
    for u in g.degreeSequence():
        if u > 1:
            i += 1
    print(i, len(g.degreeSequence()))

    folder = "Data/" + folder + "/"
    print("Saving data")
    saveData(folder, g, contentPost)

