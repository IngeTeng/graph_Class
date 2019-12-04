#迭代版bfs
import collections

def bfs(G,s):
    #P为记录每一个节点的父节点的字典
    P={s:None}
    Q=collections.deque()
    Q.append(s)
    while Q:
        u=Q.popleft()
        for v in G[u]:
            if v in P.keys():
                continue
            P[v]=P.get(v,u)
            Q.append(v)
    return P

#图的邻接字典
G={
    'a':{'b','f'},
    'b':{'c','d','f'},
    'c':{'d'},
    'd':{'e','f'},
    'e':{'f'},
    'f':{}
}

P=bfs(G,'a')
print(P)


#获取a至e的路径
u='e'
path=[u]
while P[u] is not None:
    path.append(P[u])
    u=P[u]
path.reverse()
print(path)