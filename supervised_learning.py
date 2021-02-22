from util import getSpotifyDataset, getWineDataset
from DT import DT
from KNN import KNN
from SVM import SVM
from BDT import BDT
from NN import NN

def RunSpotifyData():
    X,y = getSpotifyDataset()
    print ("----------DT------------")
    print()
    DT(X, y, 0.7, "Spotify Dataset")
    print ()
    print ("----------NN------------")
    print()
    NN(X, y, 0.7, "Spotify Dataset")
    print ("----------BDT------------")
    print()
    BDT(X,y, 0.7, "Spotify Dataset")
    print ("----------SVM------------")
    print()
    SVM(X, y, "Spotify Dataset")
    print ("----------KNN------------")
    print()
    KNN(X, y, 0.7, 8, "Spotify Dataset")


def RunWineData():
    X,y = getWineDataset()
    print ("----------DT------------")
    print()
    DT(X, y, 0.7, "Wine Dataset")
    print ()
    print ("----------NN------------")
    print()
    NN(X, y, 0.7, "Wine Dataset")
    print ("----------BDT------------")
    print()
    BDT(X,y, 0.7, "Wine Dataset")
    print ("----------SVM------------")
    print()
    SVM(X, y, "Wine Dataset")
    print ("----------KNN------------")
    print()
    KNN(X, y, 0.7, 5, "Wine Dataset")

RunSpotifyData()
RunWineData()