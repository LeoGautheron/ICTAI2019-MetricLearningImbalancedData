#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import time

import numpy as np

f = "../2018-04-03_datasets/"


def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


def loadAbalone():
    data, n, d = loadCsv(f + 'abalone/abalone.data')

    sex = data[:, 0]
    isMale = (sex == "M").astype(float).reshape(-1, 1)
    isFemale = (sex == "F").astype(float).reshape(-1, 1)
    isInfant = (sex == "I").astype(float).reshape(-1, 1)
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawX = np.hstack((isMale, isFemale, isInfant, rawX))
    rawY = data[:, d-1].astype(int)

    rawY[rawY != 4] = -1
    rawY[rawY == 4] = 1
    return rawX, rawY


def loadAustralian():
    data, n, d = loadCsv(f + 'australian/australian.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadAutompg():
    data, n, d = loadCsv(f + 'autompg/autompg.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadBalance():
    data, n, d = loadCsv(f + 'balance/balance.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0]
    rawY = rawY.astype(np.dtype(('U10', 1)))
    rawY[rawY != 'L'] = "-1"
    rawY[rawY == 'L'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadBreast():
    data, n, d = loadCsv(f + 'breast/breast-cancer-wisconsin.data')
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 4] = -1
    rawY[rawY == 4] = 1
    return rawX, rawY


def loadBupa():
    data, n, d = loadCsv(f + 'bupa/bupa.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadCovetype():
    data, n, d = loadCsv(f + 'covtype/covtype.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 4] = -1
    rawY[rawY == 4] = 1
    return rawX, rawY


def loadCreditcard():
    data = np.load(f + 'creditcard/creditcard.npy')
    n, d = data.shape
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY == 0] = -1
    return rawX, rawY


def loadCreditcardoneday():
    data = np.load(f + 'worldline/oneday.npy')
    n, d = data.shape
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY == 0] = -1
    return rawX, rawY


def loadCreditcardsmall():
    data = np.load(f + 'worldline/small.npy')
    n, d = data.shape
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY == 0] = -1
    return rawX, rawY


def loadEcoli():
    data, n, d = loadCsv(f + 'ecoli/ecoli.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'cp'] = '-1'
    rawY[rawY == 'cp'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadGerman():
    data, n, d = loadCsv(f + 'german/german.data')
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadGlass():
    data, n, d = loadCsv(f + 'glass/glass.data')
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadHaberman():
    data, n, d = loadCsv(f + 'haberman/haberman.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    return rawX, rawY


def loadHayes():
    data, n, d = loadCsv(f + 'hayes/hayes-roth.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 3] = -1
    rawY[rawY == 3] = 1
    return rawX, rawY


def loadHeart():
    data, n, d = loadCsv(f + 'heart/heart.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadIonosphere():
    data, n, d = loadCsv(f + 'ionosphere/ionosphere.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'b'] = '-1'
    rawY[rawY == 'b'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadIris():
    data, n, d = loadCsv(f + 'iris/iris.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != "Iris-virginica"] = '-1'
    rawY[rawY == "Iris-virginica"] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadIsolet():
    data, n, d = loadCsv(f + 'isolet/isolet.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 6] = -1
    rawY[rawY == 6] = 1
    return rawX, rawY


def loadLetter():
    data, n, d = loadCsv(f + 'letter/letter.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0]

    rawY[rawY == "A"] = '1'
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadLibras():
    data, n, d = loadCsv(f + 'libras/libras.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)

    rawY[rawY != 1] = -1
    return rawX, rawY


def loadNewthyroid():
    data, n, d = loadCsv(f + 'newthyroid/newthyroid.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY < 2] = -1
    rawY[rawY >= 2] = 1
    return rawX, rawY


def loadPageblocks():
    data, n, d = loadCsv(f + 'pageblocks/pageblocks.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]

    rawY = rawY.astype(int)
    rawY[rawY == 1] = -1
    rawY[rawY == 2] = -1
    rawY[rawY == 3] = 1
    rawY[rawY == 4] = 1
    rawY[rawY == 5] = 1
    return rawX, rawY


def loadPima():
    data, n, d = loadCsv(f + 'pima/pima-indians-diabetes.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadRing():
    data, n, d = loadCsv(f + 'ring/ring.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY == 1] = -1
    rawY[rawY == 0] = 1
    return rawX, rawY


def loadSatimage():
    data, n, d = loadCsv(f + 'satimage/satimage.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)

    rawY[rawY != 4] = -1
    rawY[rawY == 4] = 1
    return rawX, rawY


def loadSegmentation():
    data, n, d = loadCsv(f + 'segmentation/segmentation.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0]

    rawY[rawY == "WINDOW"] = '1'
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadShuttle():
    data, n, d = loadCsv(f + 'shuttle/shuttle.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]

    rawY = rawY.astype(int)
    rawY[rawY != 3] = -1
    rawY[rawY == 3] = 1

    return rawX, rawY


def loadSonar():
    data, n, d = loadCsv(f + 'sonar/sonar.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'R'] = '-1'
    rawY[rawY == 'R'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadSpambase():
    data, n, d = loadCsv(f + 'spambase/spambase.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadSpectfheart():
    data, n, d = loadCsv(f + 'spectfheart/spectfheart.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY == 1] = -1
    rawY[rawY == 0] = 1
    return rawX, rawY


def loadSplice():
    data, n, d = loadCsv(f + 'splice/splice.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0].astype(int)
    rawY[rawY == 1] = 2
    rawY[rawY == -1] = 1
    rawY[rawY == 2] = -1
    return rawX, rawY


def loadVehicle():
    data, n, d = loadCsv(f + 'vehicle/vehicle.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != "van"] = '-1'
    rawY[rawY == "van"] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadWdbc():
    data, n, d = loadCsv(f + 'wdbc/wdbc.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'M'] = '-1'
    rawY[rawY == 'M'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadWine():
    data, n, d = loadCsv(f + 'wine/wine.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadYeast():
    data, n, d = loadCsv(f + 'yeast/yeast.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY == "ME3"] = '1'
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadMnist():
    # General code to load train and test data from mnist
    import struct
    import gzip
    X = []  # two elements: first array of train, then array of test
    Y = []  # same
    for filename in [f + "mnist/train-labels-idx1-ubyte.gz",
                     f + "mnist/t10k-labels-idx1-ubyte.gz"]:
        with gzip.open(filename) as file:
            struct.unpack(">II", file.read(8))
            Y.append(np.fromstring(file.read(), dtype=np.int8))
    for (i, filename) in enumerate([f + "mnist/train-images-idx3-ubyte.gz",
                                    f + "mnist/t10k-images-idx3-ubyte.gz"]):
        with gzip.open(filename) as file:
            _, _, rows, cols = struct.unpack(">IIII", file.read(16))
            images = np.fromstring(
                    file.read(), dtype=np.uint8).reshape(len(Y[i]), rows*cols)
            images = images.astype(float)
            X.append(images)

    # Custom code to merge train and test, and select the
    # class 1 as positive class
    X = np.vstack((X[0], X[1]))
    Y = np.hstack((Y[0], Y[1]))

    """
    positiveClass = 0
    negativeClass = 6
    X = X[np.logical_or(Y == positiveClass, Y == negativeClass)]
    Y = Y[np.logical_or(Y == positiveClass, Y == negativeClass)]
    Y[Y != positiveClass] = -1
    Y[Y == positiveClass] = 1
    """

    from sklearn.preprocessing import scale
    X = scale(X)

    return X, Y


d = {}
s = time.time()

d["abalone"] = loadAbalone()
# d["australian"] = loadAustralian()
# d["autompg"] = loadAutompg()
# d["balance"] = loadBalance()
# d["breast"] = loadBreast()
# d["bupa"] = loadBupa()
# d["covtype"] = loadCovetype()
# d["creditcard"] = loadCreditcard()
# d["creditcardoneday"] = loadCreditcardoneday()
# d["creditcardsmall"] = loadCreditcardsmall()
# d["ecoli"] = loadEcoli()
# d["german"] = loadGerman()
# d["glass"] = loadGlass()
# d["haberman"] = loadHaberman()
# d["hayes"] = loadHayes()
# d["heart"] = loadHeart()
# d["iono"] = loadIonosphere()
# d["iris"] = loadIris()
# d["isolet"] = loadIsolet()
# d["letter"] = loadLetter()
# d["libras"] = loadLibras()
# d["newthyroid"] = loadNewthyroid()
# d["pageblocks"] = loadPageblocks()
# d["pima"] = loadPima()
# d["ring"] = loadRing()
# d["satimage"] = loadSatimage()
# d["segmentation"] = loadSegmentation()
# d["shuttle"] = loadShuttle()
# d["sonar"] = loadSonar()
# d["spambase"] = loadSpambase()
# d["spectfheart"] = loadSpectfheart()
# d["splice"] = loadSplice()
# d["vehicle"] = loadVehicle()
# d["wdbc"] = loadWdbc()
# d["wine"] = loadWine()
# d["yeast"] = loadYeast()

# d["mnist"] = loadMnist()

print("Data loaded in {:5.2f}".format(time.time()-s))
