from sklearn.neural_network import MLPClassifier

class MLPClassify(object):
    A1 = []
    B1 = []
    C1 = []
    D1 = []
    E1 = []    
    F1 = []
    A2 = []
    B2 = []
    C2 = []
    D2 = []
    E2 = []    
    F2 = []
    A3 = []
    B3 = []
    C3 = []
    D3 = []
    E3 = []    
    F3 = []
    
    def __init__(self):
        self.A1 = self.readdata("data/EMG/rotation1/class1.txt") 
        self.B1 = self.readdata("data/EMG/rotation1/class2.txt") 
        self.C1 = self.readdata("data/EMG/rotation1/class3.txt") 
        self.D1 = self.readdata("data/EMG/rotation1/class4.txt") 
        self.E1 = self.readdata("data/EMG/rotation1/class5.txt")
        self.F1 = self.readdata("data/EMG/rotation1/class6.txt") 
        
        self.A2 = self.readdata("data/EMG/rotation2/class1.txt") 
        self.B2 = self.readdata("data/EMG/rotation2/class2.txt") 
        self.C2 = self.readdata("data/EMG/rotation2/class3.txt") 
        self.D2 = self.readdata("data/EMG/rotation2/class4.txt") 
        self.E2 = self.readdata("data/EMG/rotation2/class5.txt")
        self.F2 = self.readdata("data/EMG/rotation2/class6.txt") 
        
        self.A3 = self.readdata("data/EMG/rotation3/class1.txt") 
        self.B3 = self.readdata("data/EMG/rotation3/class2.txt") 
        self.C3 = self.readdata("data/EMG/rotation3/class3.txt") 
        self.D3 = self.readdata("data/EMG/rotation3/class4.txt") 
        self.E3 = self.readdata("data/EMG/rotation3/class5.txt")
        self.F3 = self.readdata("data/EMG/rotation3/class6.txt")       
 
    def readdata(self, filename1):
        fp1 = open(filename1)
       
        pe =[]

        for line1 in fp1:
            xi = []
            strlist1 = line1.split(",")          
            for i in range(len(strlist1)):
                xi.append(float(strlist1[i]) )
            pe+= [xi]            
        return pe 
    
    def selectfeature(self, data, select):
        newdata = []
        for line in data:
            aline = []
#             print(len(line))
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
    
    def gettraindata(self, pe, v):
        sx=[]
        sy=[]
        for i in range(len(pe)):
            sy+= [v]
            sx+= [pe[i]]
        return sx, sy
    
    
    def gettestdata(self, pe1, pe2, v):
        tx=[]
        ty=[]
        for i in range(len(pe1)):
                ty+= [v]
                tx+= [pe1[i]]
        for j in range(len(pe2)):
                ty+= [v]
                tx+= [pe2[j]]
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
        newA1 = self.selectfeature(self.A1, select)
        newB1 = self.selectfeature(self.B1, select)
        newC1 = self.selectfeature(self.C1, select)
        newD1 = self.selectfeature(self.D1, select)
        newE1 = self.selectfeature(self.E1, select)
        newF1 = self.selectfeature(self.F1, select)       
        
        newA2 = self.selectfeature(self.A2, select)
        newB2 = self.selectfeature(self.B2, select)
        newC2 = self.selectfeature(self.C2, select)
        newD2 = self.selectfeature(self.D2, select)
        newE2 = self.selectfeature(self.E2, select)
        newF2 = self.selectfeature(self.F2, select)
        
        newA3 = self.selectfeature(self.A3, select)
        newB3 = self.selectfeature(self.B3, select)
        newC3 = self.selectfeature(self.C3, select)
        newD3 = self.selectfeature(self.D3, select)
        newE3 = self.selectfeature(self.E3, select)
        newF3 = self.selectfeature(self.F3, select)
        
        sax, say = self.gettraindata(newA2, 1)
        sbx, sby = self.gettraindata(newB2, 2)
        scx, scy = self.gettraindata(newC2, 3)
        sdx, sdy = self.gettraindata(newD2, 4)
        sex, sey = self.gettraindata(newE2, 5)
        sfx, sfy = self.gettraindata(newF2, 6)
        
        sx = sax + sbx + scx + sdx + sex + sfx
        sy = say + sby + scy + sdy + sey + sfy
        
                   
        clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
            hidden_layer_sizes=(100,70), random_state=1)
        
        clf.fit(sx, sy)
        
        tax, tay = self.gettestdata(newA1, newA3, 1)
        tbx, tby = self.gettestdata(newB1, newB3, 2)
        tcx, tcy = self.gettestdata(newC1, newC3, 3)
        tdx, tdy = self.gettestdata(newD1, newD3, 4)
        tex, tey = self.gettestdata(newE1, newE3, 5)
        tfx, tfy = self.gettestdata(newF1, newF3, 6)

        tx = tax + tbx + tcx + tdx + tex + tfx
        ty = tay + tby + tcy + tdy + tey + tfy
       
        score = clf.score(tx, ty)
        
        return score
