
import os
tmp=os.listdir('sighan2/pair_data/simplified')
print(tmp)
# with open('sighan2/pair_data/simplified') as f:
#     pass


from pathlib import Path
output_filedir = Path(__file__).resolve().parent/ 'sighan2/pair_data/simplified' # 获取绝对路径的方法

print(output_filedir)
listcorrect=[]
listerror=[]

print('111111111111111')
tmp.sort()
for i in  tmp:
    now=output_filedir/i
    with open(now) as f:
        if 'correct' in i:
            listcorrect+=[j.strip() for j in f.readlines()]
        else:
            listerror += [j.strip() for j in f.readlines()]
    print(now)


print(1111111)


with open('正确预料中文' ,'w') as f, open('错误预料中文','w') as g:
    f.writelines([i+'\n' for i in listcorrect])
    g.writelines([i+'\n' for i in listerror])
print('over')