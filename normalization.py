
filename1 = "data/SetEfeastrures"
filename2 = "data/NEfeature"

fp1 = open(filename1)
fp2 = open(filename2,"w+") 
      
pe =[]

max = []
min = []

for i in range(0,6):
    max.append(0)
    min.append(2)

for line1 in fp1:
    xi = []
    strlist1 = line1.split(",")          
    for i in range(len(strlist1)):
        t = float(strlist1[i])
        xi.append(t)
        if(max[i]<t): max[i]=t
        if(min[i]>t): min[i]=t
    pe+= [xi]            

for i in range(0,100):
    out=""
    for j in range(0,6):
        if(max[j]>min[j]):
            m= (pe[i][j]-min[j])/(max[j]-min[j])
            out += (str(m)+",")
        else:
            out += (str(pe[i][j])+",")
    out = out[:-1]
    out += "\n"
    fp2.write(out);  

fp1.close()
fp2.close()      