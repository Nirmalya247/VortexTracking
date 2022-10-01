from re import T
import meshio
import numpy as np
import math

import copy
from scipy.optimize import linear_sum_assignment
from itertools import combinations

xn = 192
yn = 64
zn = 48

fileReadPath = 'VortexStreet3DOWC/'
fileSavePath = 't3/'

from_time = 127
to_time = 200
threshold = 0.17

current_timestep = [0]
seeds = {}

# ------------
def getData(meshData, x, y, z):
    return meshData.point_data['magnitude'][x + y * 192 + z * 192 * 64][0]

def prepData(meshData):
    d = np.zeros((xn, yn, zn))
    for x in range(0, xn):
        for y in range(0, yn):
            for z in range(0, zn):
                d[x, y, z] = getData(meshData, x, y, z)
    return d

def checkBounds(x, y, z, d):
    if x >= 0 and y >= 0 and z >= 0 and x < xn and y < yn and z < zn and d[x, y, z] >= threshold:
        return True
    return False

def traverse(X, Y, Z, d, lst):
    Q = []
    Q.append([X, Y, Z])
    while(len(Q) > 0):
        cur = Q.pop()
        for z in [0, -1, 1]:
            for y in [0, -1, 1]:
                for x in [0, -1, 1]:
                    if(checkBounds(cur[0] + x, cur[1] + y, cur[2] + z, d)):
                        lst.append([(cur[0] + x) + (cur[1] + y) * xn + (cur[2] + z) * (xn * yn)])

                        d[cur[0] + x, cur[1] + y, cur[2] + z] = -1
                        Q.append([cur[0] + x, cur[1] + y, cur[2] + z])

def powerset(lst):
    pst = set()
    for i in range(len(lst)+1):
        for element in combinations(lst,i):
            pst.add(element)
    return pst

def saveVTK(files, mesh, clique, comp, d):
    data = "# vtk DataFile Version 2.0\nVolume example\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 192 64 48\nASPECT_RATIO 1 1 1\nORIGIN 0 0 0\nPOINT_DATA 589824\nSCALARS feature_scalar float 1\nLOOKUP_TABLE default\n"
    tD = ["0.00" for i in range(xn * yn * zn)]
    tN = 0
    minV = np.min(d)
    maxV = np.max(d)
    for z in range(zn):
        for y in range(yn):
            for x in range(xn):
                tD[tN] = "{:.2f}".format((d[x, y, z] - minV) / (maxV - minV))
                tN = tN + 1
    for c in clique:
      for p in comp[c]:
        x = int(p[0]) % xn
        y = int(p[0] / xn) % yn
        z = int(p[0] / (xn * yn))
        try:
          tD[int(p[0])] = "{:.2f}".format(color_arr[c] + (d[x, y, z] - minV) / (maxV - minV))
        except KeyError:
          tD[int(p[0])] = "{:.2f}".format((d[x, y, z] - minV) / (maxV - minV))
    
    data = data + "\n".join(tD)
    f = open(fileSavePath + 'Square_t_' + str(files) + '.vtk', "w")
    f.write(data)
    f.close()
# ------------

color_arr = {3: 2, 2: 2, 0: 1, 1: 1}
for files in range(from_time, to_time):
    mesh1 = meshio.read(fileReadPath + 'SquareCylinderOkuboWeiss_t_' + str(files) + '.vtk')
    mesh2 = meshio.read(fileReadPath + 'SquareCylinderOkuboWeiss_t_' + str(files + 1) + '.vtk')

    d1 = prepData(mesh1)
    d2 = prepData(mesh2)

    # components ------------------------
    components1 = []
    components2 = []

    td1 = d1.copy()
    td2 = d2.copy()

    for z in range(zn):
        for y in range(yn):
            for x in range(xn):
                if(td1[x, y, z] > threshold):
                    lst = []
                    traverse(x, y, z, td1, lst)
                    components1.append(lst)
                if(td2[x, y, z] > threshold):
                    lst = []
                    traverse(x, y, z, td2, lst)
                    components2.append(lst)
    
    comp1 = []  
    comp2 = []

    for i in components1:
        comp1.append(np.array(i))
    for i in components2:
        comp2.append(np.array(i))

    # create graph1 ------------------------
    graph = np.zeros([len(comp1), len(comp2)])
    for i in range(len(comp1)):
        for j in range(len(comp2)):
            common = np.intersect1d(comp1[i], comp2[j])
            graph[i][j] = len(common) / max(len(comp1[i]), len(comp2[j]))
    
    # min cost bipertite
    row_ind, col_ind = linear_sum_assignment(graph, maximize=True)

    u = [i for i in range(len(comp1))]
    v = [i for i in range(len(comp2))]

    numRows = len(comp1)
    numCols = len(comp2)

    cliques = []
    clique_dict = {}
    nodes_string = []
    nodes_list = []
    nodes_set = set()

    for i in range(len(row_ind)):
        c = set()

        u_fixed = row_ind[i]
        v_fixed = col_ind[i]

        # removed edges for u_fixed and v_fixed
        u_rem = [j for j in range(numRows) if (u_fixed != j and graph[j][v_fixed] > 0)]
        v_rem = [j for j in range(numCols) if (v_fixed != j and graph[u_fixed][j] > 0)]

        u_rem_pow = powerset(u_rem)
        v_rem_pow = powerset(v_rem)

        # c += {((u_fix, u_power_element_0), (v_fix)), ....}
        for p in u_rem_pow:
            curr = [(u_fixed,) + p, (v_fixed,)]
            c.add(tuple(curr))
        # c += {((v_fix, v_power_element_0), (u_fix)), ....}
        for q in v_rem_pow:
            curr = [(u_fixed,), (v_fixed,) + q]
            c.add(tuple(curr))
        
        tt = set()
        for i in c:
            nodes_set.add(i)
            temp = []
            temp.append(list(i[0]))
            temp.append(list(i[1]))
            nodes_list.append(temp)
            clique_dict[str(temp)] = len(cliques)
            nodes_string.append(str(temp))
            tt.add(str(temp))
        cliques.append(tt)
    
    # graph pf cliques ------------------------
    graph2 = np.zeros([len(nodes_string),len(nodes_string)])

    for i in range(len(nodes_string)):
        for j in range(i,len(nodes_string)):
            if(clique_dict[nodes_string[i]] != clique_dict[nodes_string[j]]):
                u1, v1 = nodes_list[i]
                u2, v2 = nodes_list[j]

                u1 = np.array(u1)
                v1 = np.array(v1)
                u2 = np.array(u2)
                v2 = np.array(v2)

                if(len(np.intersect1d(u1,u2)) > 0 or len(np.intersect1d(v1,v2)) > 0):
                    graph2[i][j] = 1
                    graph2[j][i] = 1

    # node index
    node_indexes = {}
    for i in nodes_string:
        node_indexes[i] = len(node_indexes)
    
    # node len
    numNodes = len(nodes_string)

    # graph2 node weights
    node_weights = []
    for i in range(numNodes):
        u = nodes_list[i][0]
        v = nodes_list[i][1]

        # sum(|intersect(ui, v)|)/|v|
        if(len(u) == 1):
            # split
            numerator = 0
            denominator = len(comp1[u[0]])
            for vi in v:
                numerator += len(np.intersect1d(comp1[u[0]], comp2[vi]))
                denominator = max(denominator, len(comp2[vi]))
            node_weights.append(numerator / denominator)
        # sum(|intersect(vi, u)|)/|u|
        else:
            # merge
            numerator = 0
            denominator = len(comp2[v[0]])
            for ui in u:
                numerator += len(np.intersect1d(comp1[ui], comp2[v[0]]))
                denominator = max(denominator, len(comp1[ui]))
            node_weights.append(numerator / denominator)

    # graph2 clique weights, updates cliques[i] = {sorted(nodes) : 16}
    clique_weights = []
    for c in range(len(cliques)):
        temp = []
        for i in cliques[c]:
            temp.append([node_weights[node_indexes[i]], node_indexes[i]])
        temp.sort(reverse=True)
        if(len(temp) > 16):
            temp = temp[:16]
            newNodes = set()
            for x in temp:
                newNodes.add(nodes_string[x[1]])
            cliques[c] = newNodes
        clique_weights.append(temp)
    
    # differences of clique weights => 1st node - 2nd node, index
    differences = []
    for c in clique_weights:
        if(len(c) == 1):
            differences.append([c[0][0], len(differences)])
        else:
            differences.append([c[0][0] - c[1][0], len(differences)])
    differences.sort(reverse=True)

    # sorted cliques based on differences
    cliques_sorted = []
    for i in differences:
        cliques_sorted.append(list(cliques[i[1]]))
    
    # clique len
    numCliques = len(cliques)


    # did not understand
    abc = []
    temp = [0 for i in range(numCliques)]
    abc.append(temp)
    for i in range(numCliques):
        for j in range(len(cliques_sorted[i])-1):
            for k in range(len(abc)):
                #print('!',end='')
                temp = copy.deepcopy(abc[k])
                temp[i] = j+1
                abc.append(temp)
   
    max_score = 0
    max_combination = -1

    for i in range(len(abc)):
        score = 0
        comb = abc[i]
        for j in range(len(comb)):
            for k in range(0,j):
                if(graph2[node_indexes[cliques_sorted[k][comb[k]]]][node_indexes[cliques_sorted[j][comb[j]]]]):
                    # not a valid choice
                    score = -9999999999
            score += node_weights[node_indexes[cliques_sorted[j][comb[j]]]]
        #print(score)
        if(score > max_score):
            max_score = score
            max_combination = i
    
    explaination = []
    for i in range(len(abc[max_combination])):
        explaination.append(cliques_sorted[i][abc[max_combination][i]])
    print('T = ',files,' to ','T = ',files+1)

    tCliques = []
    for i in explaination:
      tNode = nodes_list[node_indexes[i]]
      u = tNode[0]
      v = tNode[1]
      for ui in u:
        tCliques.append(ui)
      print(tNode)
    
    newColor = { }
    
    if len(color_arr) < 2:
      for i in explaination:
        tNode = nodes_list[node_indexes[i]]
        u = tNode[0]
        for ui in u:
          if len(color_arr) < 2:
            try:
              color_arr[ui]
            except KeyError:
              print('adding color from v')
              tV = list(set([1, 2]) - set(color_arr.values()))[0]
              color_arr[ui] = tV
              newColor[ui] = tV
    else:
      for i in explaination:
        tNode = nodes_list[node_indexes[i]]
        u = tNode[0]
        v = tNode[1]
        for vi in v:
            try:
              newColor[vi] = color_arr[u[0]]
            except KeyError:
              print('* color not found ', u[0])
      

    if len(color_arr) < 2:
      for i in explaination:
        tNode = nodes_list[node_indexes[i]]
        v = tNode[1]
        for vi in v:
          if len(color_arr) < 2:
            try:
              color_arr[vi]
            except KeyError:
              print('adding color from u')
              newColor[vi] = list(set([1, 2]) - set(color_arr.values()))[0]
    else:
      for i in explaination:
        tNode = nodes_list[node_indexes[i]]
        u = tNode[0]
        v = tNode[1]
        for vi in v:
            try:
                newColor[vi] = color_arr[u[0]]
            except KeyError:
                print('* color not found ', u[0])
    
    print('colors: ', color_arr, ' features: ', tCliques)
    print('next color: ', newColor)
    print('------------------------')
    saveVTK(files, mesh1, tCliques, comp1, d1)

    # for i in explaination:
    #   tNode = nodes_list[node_indexes[i]]
    #   u = tNode[0]
    #   v = tNode[1]
    #   if len(v) == 1:
    #     for ui in list(set(u) - set(v)):
    #       del color_arr[ui]
    color_arr = newColor

print('done')