import pylab as pl
import time
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.dates as mdates
import neurolab as nl

start_time = time.time()

def bytedate2num(fmt):
    def converter(b):
        return mdates.strpdate2num(fmt)(b.decode('ascii'))
    return converter

date_converter = bytedate2num("%Y-%m-%d")

# Main Program Here
print("Initiating Program...")
f = open('result', 'w')
data = np.recfromtxt('sp500.csv',
                     delimiter=',',
                     # names =('time', 'price'))
                     names =('time','price'),
                     converters={'time': date_converter})

# NORMASLISASI DATA
min = 10000
rangeMax = 1
rangeMin = -1
max = 0
nI = []
d = []
for i in range (0, data.__len__()):
    if data.price[i] < min:
         min = data.price[i]
    if data.price[i] > max:
         max = data.price[i]
for i in range(0, data.__len__()):
    d.append((data.price[i] - min) / (max- min) * (rangeMax - rangeMin) + rangeMin)

# prevDataoptions = [1,3,5]
# nHiddenoptions = [1,3,5]
# nTrainingDataOptions = [500]
# learning_rateOptions = [0.01, 0.03, 0.07]
prevDataoptions = [5]
nHiddenoptions = [5]
nTrainingDataOptions = [300]
learning_rateOptions = [0.03]
counterProgram = 0

for s0 in range(0, prevDataoptions.__len__()):
    prevData = prevDataoptions[s0]
    # DATA TEST
    test = []
    answer = []
    it = 750
    cc = 0
    nTest = 300
    while (cc < nTest):
        test.append([])
        answer.append([])
        for j in range(cc+it, cc+it+prevData+1):
            test[cc].append(d[j])
        answer[cc].append(d[j])
        test[cc].pop()
        it+=1
        cc+=1
    for s1 in range(0,nHiddenoptions.__len__()):
        nH = nHiddenoptions[s1]
        for s2 in range(0, nTrainingDataOptions.__len__()):
            nTraining = nTrainingDataOptions[s2]
            for s3 in range(0, learning_rateOptions.__len__()):
                er = 0
                learning_rate = learning_rateOptions[s3]
                counterProgram+=1
                avMAPE = 0
                for experimentNumber in range (0, 30):
                    # MEMBUAT JARINGAN
                    net = nl.net.newff([[-1, 1]]*prevData, [nH, 1])

                    # print("Initiating The Training Process...")
                    input = []
                    target = []
                    i = 0
                    while (i < nTraining):
                        input.append([])
                        target.append([])
                        for j in range(i, i+prevData+1):
                            input[i].append(d[j])
                        target[i].append(d[j])
                        input[i].pop()
                        i += 1

                    error = nl.train.train_rprop(net, input, target, show=100, epochs=nTraining, lr=learning_rate)

                    er += error[-1]

                    out = net.sim(test)

                    actAnswer = []
                    actOut = []
                    MAPE = 0.00000000000001
                    for i in range (0, out.__len__()):
                        #   actual data
                        actAnswer.append((answer[i][0]-rangeMin) / (rangeMax-(rangeMin)) * (max-min) + min)
                        actOut.append((out[i][0]-(rangeMin)) / (rangeMax-(rangeMin)) * (max-min) + min)
                        MAPE += abs( ((actAnswer[i] - actOut[i]) / actAnswer[i]) )
                        # print(actAnswer[i]," malah ", actOut[i])
                    MAPE -= 0.00000000000001
                    MAPE = MAPE / nTest * 100.00
                    avMAPE += MAPE
                print(counterProgram,";",prevData,";",nH,";",learning_rate,";", er/30,";",avMAPE/30,"%")
                f.write(str(counterProgram))
                f.write(";")
                f.write(str(prevData))
                f.write(";")
                f.write(str(nH))
                f.write(";")
                f.write(str(learning_rate))
                f.write(";")
                f.write(str(er/30))
                f.write(";")
                f.write(str(avMAPE/30))
                f.write("\n")

pl.subplot(211)
pl.title("Error Rate")
pl.xlabel("Number of Epochs")
pl.ylabel("Error Rate")
pl.plot(error)

pl.subplot(212)
pl.xlabel("Date")
pl.ylabel("Price Rate")
act, = pl.plot(actAnswer, "g-", label="Actual")
pdc, = pl.plot(actOut, "r-", label="Predicted")
pl.legend([act,pdc],["actual","predicted"])
plot.grid(True)
plot.show()
print("Done in ",time.time()-start_time, " seconds.")
f.close()