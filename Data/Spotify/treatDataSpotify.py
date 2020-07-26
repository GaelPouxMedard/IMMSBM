import sys
import TP1
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials


cid = '0d90192fa6d74cacabfed3550b454c83'
secret = 'e05850cb685a49fd82b6ae5038bd2052'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



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

            with open(folder+"graph.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in g.graphDict:
                    for v in g.graphDict[u]:
                        f.write(str(v)+"\t"+str(u)+"\n")


                break

        except:
            print("Retrying to write file")
            pass


def retreatCorpus(songs):
    nbPlaylists=2000
    tags = "english,rock"
    lgWindow=3

    g={}
    contentPosts={}

    borneSup=nbPlaylists//50 + 1
    iter=0
    tracksTot=0
    for page in range(0, borneSup):
        query=sp.search(q=tags, type="playlist", limit=50, offset=50*page)
        for playlist in query["playlists"]["items"]:
            if iter>nbPlaylists:
                break
            iter+=1
            if iter%5==0:
                print("Playlist", iter, "/", nbPlaylists)
            print(playlist["name"], playlist["tracks"], playlist["id"])
            playlistUsr = sp.user_playlist(playlist["owner"]["id"], playlist["id"])["tracks"]["items"]

            if len(playlistUsr)<lgWindow:
                continue  # On doit definir une fenetre dans laquelle on considere que les musiques ont une influence
            nbTrack=0
            for track_i in range(len(playlistUsr)):
                try:
                    track=playlistUsr[track_i]
                    nameSongs = track["track"]["name"].replace(" ", "_").lower()
                    artistsSongs = []
                    for art in track["track"]["artists"]:
                        artistsSongs.append(art["name"].replace(" ", "_").lower())

                    time = track["added_at"]
                    nbTrack+=1
                    tracksTot+=1
                    
                    if songs:
                        artists=[nameSongs]
                    else:
                        artists=artistsSongs

                    for name in artists:
                        try:
                            contentPosts[-tracksTot].append(name)
                        except:
                            contentPosts[-tracksTot] = [name]

                    try:  # Les +- tracktot c'est un moyen de simuler les couples message-reponse ; +tracksTot=message ; -tracksTot=reponse
                        g[tracksTot].append(-tracksTot)
                    except:
                        g[tracksTot] = [-tracksTot]


                    if nbTrack>lgWindow:
                        fen=range(1, lgWindow+1)
                    else:
                        fen=range(1, nbTrack+1)
                    for w in fen:
                        txtPar = playlistUsr[track_i-w]["track"]["name"].replace(" ", "_").lower()
                        artistsPar=[]
                        for art in playlistUsr[track_i-w]["track"]["artists"]:
                            artistsPar.append(art["name"].replace(" ", "_").lower())
                            
                        if songs:
                            artistsPar=[txtPar]
                        else:
                            artistsPar=artistsPar

                        for name in artistsPar:
                            try:
                                contentPosts[tracksTot].append(name)
                            except:
                                contentPosts[tracksTot]=[name]

                except Exception as e:
                    print(e)
                    pass

    return g, contentPosts



def retreat(folder, listFiles=None, songs=False):
    g, contentPost = retreatCorpus(songs=songs)

    g = TP1.Graph(g)

    folder = "Data/" + folder + "/"
    print("Saving data")
    saveData(folder, g, contentPost)