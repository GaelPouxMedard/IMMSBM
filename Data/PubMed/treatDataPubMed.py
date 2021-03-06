import io
import functools
import itertools
import gzip
import json
import TP1

import pandas
import time
import xml.etree.ElementTree as ET
import requests

def esearch_query(payload, retmax = 100, sleep=2):
    """
    Query the esearch E-utility.
    """
    url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    payload['retmax'] = retmax
    payload['retstart'] = 0
    ids = list()
    count = 1
    while payload['retstart'] < count:
        response = requests.get(url, params=payload)
        xml = ET.fromstring(response.content)
        try:
            count = int(xml.findtext('Count'))
        except:
            count=-1
        ids += [xml_id.text for xml_id in xml.findall('IdList/Id')]
        payload['retstart'] += retmax
        time.sleep(sleep)
    return ids


def harvest(folder):
    # Read MeSH terms to MeSH names
    url = 'https://raw.githubusercontent.com/dhimmel/mesh/e561301360e6de2140dedeaa7c7e17ce4714eb7f/data/terms.tsv'
    mesh_df = pandas.read_table(url)

    nameQueryDis=[]
    with open("Data/" + folder + "/Raw/NamesDiseases.txt") as f:
        for l in f:
            names, _ = l.split("\t")
            name = names.split(', ')[0]
            nameQueryDis.append(name)

    nameQuerySym=[]
    with open("Data/" + folder + "/Raw/NamesSymptoms.txt") as f:
        for l in f:
            names, _ = l.split("\t")
            name = names.split(', ')[0]
            nameQuerySym.append(name)





    print("Diseases")
    # Diseases
    disease_df=pandas.DataFrame(nameQueryDis, columns={"mesh_name"})
    bornes=[0, len(disease_df)]
    print(bornes)
    disease_df=disease_df[disease_df.index>=bornes[0]]
    disease_df=disease_df[disease_df.index<bornes[1]]

    rows_out = list()
    attemps=0

    # Query Diseases
    for i, row in disease_df.iterrows():
        while True:
            try:
                term_query = '{disease}[MeSH Major Topic]'.format(disease = row.mesh_name.lower())
                payload = {'db': 'pubmed', 'term': term_query}
                pmids = esearch_query(payload, retmax = 10000)
                row['term_query'] = term_query
                row['n_articles'] = len(pmids)
                row['pubmed_ids'] = '|'.join(pmids)
                rows_out.append(row)
                print('{} articles for {}'.format(len(pmids), row.mesh_name))

                while True:
                    try:
                        disease_pmids_df = pandas.DataFrame(rows_out)
                        print("begWrite")
                        with open("Data/" + folder + "/" + 'Raw/disease-pmids_'+str(i)+"_"+row.mesh_name.replace(" ", "_")+'.tsv', 'w') as write_file:
                            #write_file = io.TextIOWrapper(write_file)
                            #disease_pmids_df.to_csv(write_file, sep='\t', index=False)
                            txt = "mesh_name\tterm_query\tn_articles\tpubmed_ids\n"
                            txt+=row.mesh_name+"\t"+term_query+"\t"+str(len(pmids))+"\t"+'|'.join(pmids)+"\n"
                            #print(txt)
                            write_file.write(txt)

                        print("endWrite")
                        break
                    except:
                        attemps += 1
                        if attemps > 5:
                            with open("0_incorrects.txt", "a") as f:
                                f.write(row.mesh_name+"\n")
                            break
                        pass

                attemps=0
                break
            except Exception as e:
                print("==========================", e)
                attemps+=1
                if attemps>5:
                    with open("0_incorrects.txt", "a") as f:
                        f.write(row.mesh_name+"\n")
                    break
                continue

    disease_pmids_df = pandas.DataFrame(rows_out)

    with gzip.open("Data/" + folder + "/"+'Raw/disease-pmids.tsv.gz', 'w') as write_file:
        write_file = io.TextIOWrapper(write_file)
        disease_pmids_df.to_csv(write_file, sep='\t', index=False)

    '''
    print("Symptoms")
    # Symptoms

    symptom_df = pandas.DataFrame(nameQuerySym, columns={"mesh_name"})

    rows_out = list()

    # Query symptoms
    for i, row in symptom_df.iterrows():
        while True:
            try:
                term_query = '{symptom}[MeSH Terms:noexp]'.format(symptom = row.mesh_name.lower())
                payload = {'db': 'pubmed', 'term': term_query}
                pmids = esearch_query(payload, retmax = 5000, sleep=2)
                row['term_query'] = term_query
                row['n_articles'] = len(pmids)
                row['pubmed_ids'] = '|'.join(pmids)
                rows_out.append(row)
                print('{} articles for {}'.format(len(pmids), row.mesh_name))

                symptom_pmids_df = pandas.DataFrame(rows_out)
                with gzip.open("Data/" + folder + "/" + 'Raw/symptom-pmids.tsv.gz', 'w') as write_file:
                    write_file = io.TextIOWrapper(write_file)
                    symptom_pmids_df.to_csv(write_file, sep='\t', index=False)

                break
            except:
                continue

    symptom_pmids_df = pandas.DataFrame(rows_out)

    with gzip.open("Data/" + folder + "/"+'Raw/symptom-pmids.tsv.gz', 'w') as write_file:
        write_file = io.TextIOWrapper(write_file)
        symptom_pmids_df.to_csv(write_file, sep='\t', index=False)

    print(symptom_pmids_df.head())
    '''


def read_pmids_tsv(path, key, min_articles=5):
    pmids_df = pandas.read_table(path)#, compression='gzip')
    pmids_df = pmids_df[pmids_df.n_articles >= min_articles]
    return pmids_df


def saveData(folder, g, nounsPost):
    while True:
        try:
            listInfs=set()
            with open(folder + "/nounsPost.txt", "a", encoding="utf-8") as f:
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

            with open(folder+"/graph.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in g.graphDict:
                    for v in g.graphDict[u]:
                        f.write(str(v)+"\t"+str(u)+"\n")


                break

        except Exception as e:
            print("Retrying to write file :", e)
            pass


def concatenateDiseases(folder):
    from os import listdir
    listFiles = listdir("Data/" + folder + "/Raw/")
    temp = []
    for n in listFiles:
        if "disease" in n and ".tsv" in n:
            temp.append(n)
    listFiles = temp

    rows=list()
    for file in listFiles:
        with open("Data/" + folder + "/" + 'Raw/'+file, 'r') as f:
            #print(file)
            f.readline()
            r = {}
            r["mesh_name"], r["term_query"], r["n_articles"], r["pubmed_ids"] = f.readline().split("\t")
            rows.append(r)

    disease_df=pandas.DataFrame(rows)

    return disease_df


def treatHarvestedData(folder):

    symptom_df = read_pmids_tsv("Data/" + folder + "/"+'Raw/symptom-pmids.tsv.gz', key='mesh_id')
    disease_df = concatenateDiseases(folder)

    print("Gathering diseases")
    tabPublis={}
    listDis=[]
    for (dis,idPudDis) in zip(disease_df["mesh_name"], disease_df["pubmed_ids"]):
        tabIDsDis = idPudDis.split("|")
        dis=dis.split(", ")[0].replace(" ", "_")
        listDis.append(dis)
        for IDDis in tabIDsDis:
            try:
                tabPublis[IDDis].append(dis)
            except:
                tabPublis[IDDis]=[dis]


    print("Gathering symptoms")
    listSym=[]
    for (sym, idPudSym) in zip(symptom_df["mesh_name"], symptom_df["pubmed_ids"]):
        tabIDsSym = idPudSym.split("|")
        sym=sym.split(", ")[0].replace(" ", "_")
        listSym.append(sym)
        for IDSym in tabIDsSym:
            try:
                tabPublis[IDSym].append(sym)
            except:
                pass  # Si y'a aucune maladie osef des symptomes

    print("Recreate articles")
    iter=1
    g={}
    contentPost={}
    lenLoop = len(tabPublis)
    print(lenLoop)
    listInfs=set()
    for id in tabPublis:
        if iter%1000==0:
            print(iter*100./lenLoop, "%")
        try:
            g[iter].append(-iter)
        except:
            g[iter]=[-iter]

        for elem in tabPublis[id]:
            if elem in listSym:
                try:
                    contentPost[iter].append(elem)
                except:
                    contentPost[iter] = [elem]
            else:
                try:
                    contentPost[-iter].append(elem)
                except:
                    contentPost[-iter] = [elem]

        iter+=1

    return g, contentPost


def retreat(folder, listFiles=None):
    g, contentPost = treatHarvestedData(folder)

    g = TP1.Graph(g)

    folder = "Data/" + folder + "/"
    print("Saving data")
    saveData(folder, g, contentPost)

#harvest("SymptomeDisease")
#pause()

#retreat("SymptomeDisease")
