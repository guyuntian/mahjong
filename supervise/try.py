import torch
transfer = [[0,0,0,0], [1,0,0,0]]
for i in range(2, 4):
    transfer = transfer + [[i, j, 0, 0]for j in range(4, 8)]

for i in range(1, 4):
    for j in range(1, 10):
        transfer = transfer + [[4, i, j, k]for k in range(1, 5)]
for i in range(4, 11):
    transfer = transfer + [[4, i, 0, k]for k in range(1, 5)]

for i in range(1, 4):
    for j in range(1, 10):
        transfer = transfer + [[5, i, j, 1]]
for i in range(4, 11):
    transfer = transfer + [[5, i, 0, 1]]

for i in range(1, 4):
    for j in range(1, 10):
        transfer = transfer + [[6, i, j, k] for k in range(1, 4)]
for i in range(4, 11):
    transfer = transfer + [[6, i, 0, 1] for k in range(1, 4)]

for i in range(1, 4):
    for j in range(1, 10):
        transfer = transfer + [[7, i, j, k] for k in range(1, 5)]
for i in range(4, 11):
    transfer = transfer + [[7, i, 0, 1] for k in range(1, 5)]

for i in range(1, 4):
    for j in range(1, 10):
        transfer = transfer + [[8, i, j, k] for k in range(5, 10)]
for i in range(4, 11):
    transfer = transfer + [[8, i, 0, 1] for k in range(5, 10)]
print(len(transfer))

dic = torch.tensor(transfer)
print(dic)

inputs = torch.tensor([[2, 3], [0, 587]])
print(dic[inputs])