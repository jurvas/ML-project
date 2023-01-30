import sys

in_file = "clean.txt"

coding = {'A': '1', 'C': '2', 'G': '3', 'T': '4'}
labels = []

out_labels = open("labels.txt","w")
out_sequences = open("sequences.txt","w")
out_classes = open("classes.txt","w")

i = 1	 
for s in open(in_file):
    if(i%4 == 1):
        iden = s
    if(i%4 == 2):
        accen = s.split(";")
    if(i%4 == 3):
        clas = s.split(";")
    if(i%4 == 0):
        if("genus" in clas):
            seq = s.strip()
            if(len(seq)>10000):
                continue
            prem = ""
            bad = 0
            for c in seq:
                if(c!='A' and c!='C' and c!='G' and c!='T'):
                    bad = 1
                    break
                prem += coding[c]
            if(bad):
                continue
            idx = clas.index("genus")
            print(accen[idx])
            print(prem)
            
            if(accen[idx] not in labels):
                labels.append(accen[idx])
                out_classes.write(accen[idx]+"\n")
            out_labels.write(str(labels.index(accen[idx]))+"\n")
            out_sequences.write(prem+"\n")
    i += 1


out_labels.close()
out_sequences.close()
out_classes.close()
