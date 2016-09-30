from sklearn.neural_network import MLPClassifier

class MLPClassify(object):
    FA = []
    FB = []
    FC = []
    FD = []
    FE = []    
#     FF = []
    
    def __init__(self):
        
        self.FA = self.readdata("data/EEG/Z.txt","data/EEG/SetAfeastrures") 
        self.FB = self.readdata("data/EEG/O.txt","data/EEG/SetBfeastrures") 
        self.FC = self.readdata("data/EEG/N.txt","data/EEG/SetCfeastrures") 
        self.FD = self.readdata("data/EEG/F.txt","data/EEG/SetDfeastrures") 
        self.FE = self.readdata("data/EEG/S.txt","data/EEG/SetEfeastrures")
#         self.FA = self.readdata("data/NAfeature") 
#         self.FB = self.readdata("data/NBfeature") 
#         self.FC = self.readdata("data/NCfeature") 
#         self.FD = self.readdata("data/NDfeature") 
#         self.FE = self.readdata("data/NEfeature")
        print(self.FA)
 
    def readdata(self, filename1, filename2):
        fp1 = open(filename1)
        fp2 = open(filename2)
        
        pe =[]
        
        line2 = fp2.readline()
        for line1 in fp1:
            xi = []
            strlist1 = line1.split(",")                      
            for i in range(0,len(strlist1)-3):
                xi.append(float(strlist1[i]) )
            
            strlist2 = line2.split(",")                      
            for j in range(len(strlist2)):
                xi.append(float(strlist2[j]) )
            
            pe+= [xi]    
            line2 = fp2.readline()        
        return pe 
    
    def selectfeature(self, data, select):
        newdata = []
        for line in data:
            aline = []
            for i in range(len(line)):
                if select[i]==1:
                    aline.append(line[i])            
            newdata.append(aline) 
        return newdata   
    
    def setlabel(self, setlen, lable):
        sy = []
        for i in range(setlen): 
            sy+=[lable]  
        return sy
    
    def gettraindata(self, pe, l, v):
        sx=[]
        sy=[]
        for i in range(len(pe)):
            if i<l:
                sy+= [v]
                sx+= [pe[i]]
        return sx, sy
    
    
    def gettestdata(self, pe, l, v):
        tx=[]
        ty=[]
        count = 0;
        for i in range(len(pe)-1, 0, -1):
            if count<l:
                ty+= [v]
                tx+= [pe[i]]
                count = count + 1
            else:
                break
        return tx, ty
    
    
    def SenAndSpe(self, ty, pv):
        if len(ty) != len(pv):
            raise ValueError("len(ty) must equal to len(pv)")
        total_correct = total_error = 0
        TP = TN = FP = FN = 0
        for v, y in zip(pv, ty):
            if y == v:
                total_correct += 1
                if v == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                total_error += 1
                if v == 1:
                    FP += 1
                else:
                    FN += 1
        ACC = (TP+TN)*1.0/(TP+FP+FN+TN)
        SEN = (TP)*1.0/(TP+FN)
        SPC = (TN)*1.0/(TN+FP)
        return SEN, SPC, ACC


    def classify(self, select):  
#         print(select)
        
        newFA = self.selectfeature(self.FA, select)
        newFB = self.selectfeature(self.FB, select)
        newFC = self.selectfeature(self.FC, select)
        newFD = self.selectfeature(self.FD, select)
        newFE = self.selectfeature(self.FE, select)
#         newFF = self.selectfeature(self.FF, select)
        
#         if(select[6]<100):
#             select += 100;
#         traindata = select[6]/2
#         testdata = 100-traindata
        
        sax, say = self.gettraindata(newFA, 85, 1)
        sbx, sby = self.gettraindata(newFB, 85, 2)
        scx, scy = self.gettraindata(newFC, 85, 3)
        sdx, sdy = self.gettraindata(newFD, 85, 4)
        sex, sey = self.gettraindata(newFE, 85, 5)
        sx = sax + sbx + scx + sdx + sex
        sy = say + sby + scy + sdy + sey
        
        
#         n=0;
#         hide = []
#         for j in range(7,10):
#             if(select[j]>0):
#                 hide.append(select[j])
#                 n=n+1
#         
#         if(n==0):
#             return 0
#         if(n==1):    
#             clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
#                             hidden_layer_sizes=(hide[0]), random_state=1)
#         if(n==2):    
#             clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
#                             hidden_layer_sizes=(hide[0],hide[1]), random_state=1)
#         if(n==3):    
        clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
                            hidden_layer_sizes=(120,80), random_state=1)
        clf.fit(sx, sy)
        
        tax, tay = self.gettestdata(newFA, 15, 1)
        tbx, tby = self.gettestdata(newFB, 15, 2)
        tcx, tcy = self.gettestdata(newFC, 15, 3)
        tdx, tdy = self.gettestdata(newFD, 15, 4)
        tex, tey = self.gettestdata(newFE, 15, 5)
#         tfx, tfy = self.gettestdata(newFF, 20, 6)

        tx = tax + tbx + tcx + tdx + tex
        ty = tay + tby + tcy + tdy + tey
       
        score = clf.score(tx, ty)
        
        return score
