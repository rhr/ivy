import ivy
primates = ivy.tree.read(u"((((Homo:0.21,Pongo:0.21)A:0.28,Macaca:0.49)B:0.13,Ateles:0.62)C:0.38,Galago:1.00)root;")
primate_data = dict(zip(primates.tiplabels(), [0,0,1,1,0]))
