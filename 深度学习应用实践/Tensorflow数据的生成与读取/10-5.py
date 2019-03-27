import os

path = "jpg"
# 列出目录下的所有文件名称
filename = os.listdir(path)
strText = ""

with open("train_list.csv", "w") as fid:
    for a in range(len(filename)):
        print(filename[a].split("ge")[0])
        # os.sep根据系统自动采用相应的分隔符
        strText = path + os.sep + filename[a] + "," + "1" + "\n"
        fid.write(strText)
        
    fid.close()
