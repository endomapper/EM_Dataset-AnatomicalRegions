import os
import shutil

labels_train = {'rectum':0,'sigmoid':0,'descending':0,'esplenic':0,'transverse':0,
          'hepatic':0,'ascending':0,'ileocecal':0,'ileum':0,'cecum':0}
labels_test = {'rectum':0,'sigmoid':0,'descending':0,'esplenic':0,'transverse':0,
          'hepatic':0,'ascending':0,'ileocecal':0,'ileum':0,'cecum':0}
Train = ["Seq_003","Seq_011","Seq_013","Seq_093"]
Test = ["Seq_094"]
os.makedirs("Train",exist_ok=True)
os.makedirs("Test",exist_ok=True)
for label in labels_train.keys():
    os.makedirs("Train/"+label,exist_ok=True)
    os.makedirs("Test/"+label,exist_ok=True)
for Seq in [i for i in os.listdir(".") if i.startswith("Anatomical_regions")]:
    Seq_n = Seq.split("_")[-1][:-4]
    Sequence = "Seq_"+Seq_n
    print(Seq_n)
    with open(Seq,'r') as fp:
        all_lines = fp.readlines()
    for line in all_lines[1:]:
        frame_n = line.split(";")[1]
        label = line.split(";")[2]
        if Sequence in Train:
            num = labels_train[label]
            shutil.copy(Sequence+"/"+frame_n+".png","Train/"+label+"/%07d.png"%num)
            labels_train[label] = num+1
        else:
            num = labels_test[label]
            shutil.copy(Sequence+"/"+frame_n+".png","Test/"+label+"/%07d.png"%num)
            labels_test[label] = num+1
print("Se ha creado el dataset con los siguientes numeros:")
print("Train:")
print(labels_train)
print("Test:")
print(labels_test)



