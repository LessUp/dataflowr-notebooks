import numpy as np
import scipy.sparse
import sklearn.metrics

  
# graph coarsening with Heavy Edge Matching
def coarsen(A, levels,laplacian_func):
    
    graphs, parents = HEM(A, levels)
    perms = compute_perm(parents)

    laplacians = []
    for i,A in enumerate(graphs):
        M, M = A.shape
            
        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added), |E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

        L = laplacian_func(A)
        laplacians.append(L)
        
    return laplacians, perms[0] if len(perms) > 0 else None


def HEM(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the Heavy Edge Matching (HEM).

    Input
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    Output
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    Note
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape
    
    if rid is None:
        rid = np.random.permutation(range(N))
        
    ss = np.array(W.sum(axis=0)).squeeze()
    rid = np.argsort(ss)
        
        
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    print('Heavy Edge Matching coarsening with Xavier version')

    for _ in range(levels):

        # 为配对选择权重
        # weights = ones(N,1)       # metis 权重
        weights = degree            # graclus 权重
        # weights = supernode_size  # 另一种可能
        weights = np.array(weights).squeeze()

        # 配对顶点并构造根向量
        idx_row, idx_col, val = scipy.sparse.find(W)
        cc = idx_row
        rr = idx_col
        vv = val
        
        # 有待加速
        if not (list(cc)==list(np.sort(cc))):
            tmp=cc
            cc=rr
            rr=tmp

        cluster_id = HEM_one_level(cc,rr,vv,rid,weights) # cc 是有序的
        parents.append(cluster_id)

        # 计算新图的边权重
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR 更合适：row、val 对会出现多次
        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()
        
        # 把新图加入所有粗化图的列表
        graphs.append(W)
        N, N = W.shape

        # 计算度数（是否省略自环）
        degree = W.sum(axis=0)
        #degree = W.sum(axis=0) - W.diagonal()

        # 选择下一轮访问顶点的顺序
        #[~, rid]=sort(ss);     # arthur 策略
        #[~, rid]=sort(supernode_size);    #  thomas 策略
        #rid=randperm(N);                  #  metis/graclus 策略
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def HEM_one_level(rr,cc,vv,rid,weights):

    nnz = rr.shape[0]
    N = rr[nnz-1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    
                    # 第一种方法
                    if 2==1:
                        tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                    
                    # 第二种方法
                    if 1==1:
                        Wij = vv[rs+jj]
                        Wii = vv[rowstart[tid]]
                        Wjj = vv[rowstart[nid]]
                        di = weights[tid]
                        dj = weights[nid]
                        tval = (2.*Wij + Wii + Wjj) * 1./(di+dj+1e-9)
                    
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id


def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # 最后一层的顺序是随机的（由聚类算法决定）。
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:

        # 假节点排在真节点后面。
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            # 为孤立节点添加一个配套节点。
            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1

            # 在父层为孤立节点添加两个子节点。
            elif len(indices_node) is 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # 合理性检查。
    for i,indices_layer in enumerate(indices):
        M = M_last*2**i
        # 每一层缩减 2 倍（二叉树）。
        assert len(indices[0] == M)
        # 新排序没有漏掉任何索引。
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]

assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
        == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])



def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    A = A.tocoo()

    # 添加 Mnew - M 个孤立顶点。
    rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
    cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
    A = scipy.sparse.vstack([A, rows])
    A = scipy.sparse.hstack([A, cols])

    # 置换行和列。
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    assert np.abs(A - A.T).mean() < 1e-8 # 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A



def perm_data(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x

    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew))
    for i,j in enumerate(indices):
        # 已有顶点，也就是真实数据。
        if j < M:
            xnew[:,i] = x[:,j]
        # 因孤立节点而存在的假顶点。
        # 它们保持 0，这样最大池化会选择孤立节点。
        # 或者 -infty？
        else:
            xnew[:,i] = np.zeros(N)
    return xnew

