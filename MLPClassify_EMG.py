from sklearn.neural_network import MLPClassifier

class MLPClassify(object):
    FA = []
    FB = []
    FC = []
    FD = []
    FE = []    
    FF = []
    
    def __init__(self):
        
        self.FA = self.readdata("data/EMG/rotation3/class1.txt") 
        self.FB = self.readdata("data/EMG/rotation3/class2.txt") 
        self.FC = self.readdata("data/EMG/rotation3/class3.txt") 
        self.FD = self.readdata("data/EMG/rotation3/class4.txt") 
        self.FE = self.readdata("data/EMG/rotation3/class5.txt")
        self.FF = self.readdata("data/EMG/rotation3/class6.txt")       
 
    def readdata(self, filename1):
        fp1 = open(filename1)
       
        pe =[]

        for line1 in fp1:
            xi = []
            strlist1 = line1.split(",")          
            for i in range(len(strlist1)):
                xi.append(float(strlist1[i]))
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
        newFF = self.selectfeature(self.FF, select)
        
        sax, say = self.gettraindata(newFA, 4000, 1)
        sbx, sby = self.gettraindata(newFB, 4000, 2)
        scx, scy = self.gettraindata(newFC, 4000, 3)
        sdx, sdy = self.gettraindata(newFD, 4000, 4)
        sex, sey = self.gettraindata(newFE, 4000, 5)
        sfx, sfy = self.gettraindata(newFF, 4000, 6)
        
        sx = sax + sbx + scx + sdx + sex + sfx
        sy = say + sby + scy + sdy + sey + sfy
        
        
           
        clf = MLPClassifier(algorithm='l-bfgs', activation='logistic', alpha=1e-5, 
            hidden_layer_sizes=(100,70), random_state=1)
        
        clf.fit(sx, sy)
        
        tax, tay = self.gettestdata(newFA, 680, 1)
        tbx, tby = self.gettestdata(newFB, 680, 2)
        tcx, tcy = self.gettestdata(newFC, 680, 3)
        tdx, tdy = self.gettestdata(newFD, 680, 4)
        tex, tey = self.gettestdata(newFE, 680, 5)
        tfx, tfy = self.gettestdata(newFF, 680, 6)

        tx = tax + tbx + tcx + tdx + tex + tfx
        ty = tay + tby + tcy + tdy + tey + tfy
       
        score = clf.score(tx, ty)
        
        return score
