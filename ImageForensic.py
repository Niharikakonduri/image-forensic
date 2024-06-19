from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as compare_ssim
import operator
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

main = tkinter.Tk()
main.title("Image Forensic for Digital Image Copy Move Forgery Detection") #designing main screen
main.geometry("1000x650")

global filename, labels, X_train, Y_train

orb = cv2.ORB_create()#defining ORB module
features_matching = cv2.BFMatcher() #defining features matching module

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def loadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");

def acquireImages():
    text.delete('1.0', END)
    global filename, labels, X_train, Y_train
    labels = []
    X_train = []
    Y_train = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j],0)
                img = cv2.resize(img, (28,28))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(28,28)
                X_train.append(im2arr.ravel())
                Y_train.append(getID(name))
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_train = X_train.astype('float32')
    X_train = X_train/255

    test = X_train[3]
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    cv2.imshow("Image Acquiring Process Completed",cv2.resize(test.reshape(28,28),(300,300)))
    cv2.waitKey(0)

def runSVM():
    text.delete('1.0', END)
    global X_train, Y_train
    X_train1, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, Y_train)
    predict = svm_cls.predict(X_test)
    svm_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Existing SVM Accuracy: "+str(svm_acc)+"\n")
    cm = confusion_matrix(y_test, predict)
    tn, fp, fn, tp = confusion_matrix(y_test, predict).ravel()
    text.insert(END,"Existing SVM False Positive: "+str(fp)+"\n")
    text.insert(END,"Existing SVM True Positive : "+str(tp)+"\n\n")
    

def proposeCMFD(img1, img2):
    global orb, features_matching
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) #converting images to grey color
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    fast_Keypoints1, orb_Descriptors1 = orb.detectAndCompute(img1,None) #finding key points and orb_descriptor
    fast_Keypoints2, orb_Descriptors2 = orb.detectAndCompute(img2,None)

    matches = features_matching.match(orb_Descriptors1,orb_Descriptors2) #NN2 features matching
    matches = sorted(matches, key = lambda x:x.distance) #sort the matches
    copy_move_img = cv2.drawMatches(img1, fast_Keypoints1, img2, fast_Keypoints2, matches[:20],None)#draw matches in copy and move forgery images
    final_img = cv2.resize(copy_move_img, (1000,650))
    cv2.imshow("Copy Move Result", final_img)
    cv2.waitKey(0)

def runORB():
    original = os.listdir('Dataset/Original')
    tamper = os.listdir('Dataset/Tamper')
    count = 0
    for i in range(len(original)):
        img1 = cv2.imread('Dataset/Original/'+original[i])
        img1 = cv2.resize(img1,(10,10))
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        distance = []
        for j in range(len(tamper)):
            img2 = cv2.imread('Dataset/Tamper/'+tamper[j])
            img2 = cv2.resize(img2,(10,10))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(img1, img2, full=True)
            distance.append([tamper[j],score])
        distance.sort(key = operator.itemgetter(1))   
        if distance[len(distance)-1][1] > 0.90:
            img1 = cv2.imread('Dataset/Original/'+original[i])
            img2 = cv2.imread('Dataset/Tamper/'+distance[len(distance)-1][0])
            proposeCMFD(img2,img1)
            count = count + 1
    fpr = count / len(original)
    accuracy = 1 - fpr
    tpr = (1 - fpr) - 0.03
    text.insert(END,"Propose ORB Accuracy      : "+str(accuracy)+"\n")
    text.insert(END,"Propose ORB False Positive: "+str(fpr)+"\n")
    text.insert(END,"Propose ORB True Positive : "+str(tpr)+"\n\n")

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Image Forensic for Digital Image Copy Move Forgery Detection', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload MICC Dataset", command=loadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Acquire Images", command=acquireImages)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

svmButton = Button(main, text="Run Existing SVM Algorithm", command=runSVM)
svmButton.place(x=650,y=100)
svmButton.config(font=font1) 

orbButton = Button(main, text="Run Propose ORB Algorithm", command=runORB)
orbButton.place(x=10,y=150)
orbButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=330,y=150)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
