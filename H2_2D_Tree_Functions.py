########################################################################################################################
#                                                                                                                      #
#                              BLACK BOX FAST MULTIPOLE METHOD 2D                                                      #
#                                        Version 1.0                                                                   #
#                       Written for C++ by    : Sivaram Ambikasaran, Ruoxi Wang                                        #
#                       Written for Python by : Fahd Siddiqui                                                          #
#                         https://github.com/DrFahdSiddiqui/bbFMM2D-Python                      					   #
#                                                                                                                      #
# ==================================================================================================================== #
# LICENSE: MOZILLA 2.0                                                                                                 #
#   This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.                               #
#   If a copy of the MPL was not distributed with this # file, You can obtain one at http://mozilla.org/MPL/2.0/.      #
########################################################################################################################

########################################################################################################################

from math import pi
from numpy import zeros, ones, dot, append, cos, arange, kron
from H2_2D_Node import H2_2D_Node


# -------------------------------------------------------------------------------------------------------------------- #
def get_Standard_Chebyshev_Nodes(nChebNodes):
    """Obtains standard Chebyshev nodes in interval [-1,1] """
    cNode = zeros((nChebNodes, 1))
    # for k in range(0, nChebNodes):
    #     cNode[k] = -cos((k + 0.5) * pi / nChebNodes)
    cNode = cos((arange(0, nChebNodes) + .5) * pi / nChebNodes).reshape(nChebNodes, 1)
    return cNode


# -------------------------------------------------------------------------------------------------------------------- #
def get_Standard_Chebyshev_Polynomials(nChebPoly, N, x):
    """Computes the evaluation of Chebyshev polynomials at their roots"""
    T = zeros((N, nChebPoly))
    T[:, [0]] = ones([N, 1])
    if nChebPoly > 1:
        T[:, [1]] = x
        for k in range(2, nChebPoly):
            T[:, [k]] = 2.0 * x * T[:, [k - 1]] - T[:, [k - 2]]
    return T


# -------------------------------------------------------------------------------------------------------------------- #
def get_Transfer(nChebNodes, cNode, TNode):
    """Evaluates transfer from four children to parent"""
    # S = zeros([2 * nChebNodes, nChebNodes])
    S = get_Transfer_From_Parent_CNode_To_Children_CNode(nChebNodes, cNode, TNode)

    Transfer = zeros((2, nChebNodes, nChebNodes))
    Transfer[0] = S[0:nChebNodes, 0:nChebNodes]
    Transfer[1] = S[nChebNodes:2 * nChebNodes, 0:nChebNodes]

    rank = nChebNodes * nChebNodes
    # for k in range(0, 4):
    R = zeros([4, rank, rank])

    R[0] = kron(Transfer[0], Transfer[0])
    R[1] = kron(Transfer[0], Transfer[1])
    R[2] = kron(Transfer[1], Transfer[0])
    R[3] = kron(Transfer[1], Transfer[1])

    # for i in range(0, nChebNodes):
    #     for j in range(0, nChebNodes):
    #         for k in range(0, nChebNodes):
    #             for l in range(0, nChebNodes):
    #                 R[0][i * nChebNodes + j, k * nChebNodes + l] = Transfer[0][i, k] * Transfer[0][j, l]
    #                 R[1][i * nChebNodes + j, k * nChebNodes + l] = Transfer[0][i, k] * Transfer[1][j, l]
    #                 R[2][i * nChebNodes + j, k * nChebNodes + l] = Transfer[1][i, k] * Transfer[0][j, l]
    #                 R[3][i * nChebNodes + j, k * nChebNodes + l] = Transfer[1][i, k] * Transfer[1][j, l]

    return R


# -------------------------------------------------------------------------------------------------------------------- #
def get_Transfer_From_Parent_CNode_To_Children_CNode(nChebNodes, cNode, TNode):
    """Obtains interpolation operator,
    which interpolates information from Chebyshev nodes of parent to Chebyshev nodes of children"""
    childcNode = zeros((2 * nChebNodes, 1))
    childcNode[0:nChebNodes, [0]] = 0.5 * (cNode - 1)
    childcNode[nChebNodes:2 * nChebNodes, [0]] = 0.5 * (cNode + 1)
    Transfer = get_Standard_Chebyshev_Polynomials(nChebNodes, 2 * nChebNodes, childcNode)
    Transfer = (2.0 * dot(Transfer, TNode.T) - 1) / nChebNodes
    return Transfer


# -------------------------------------------------------------------------------------------------------------------- #
def get_Center_Radius(location):
    """Computes the center and radius of the smallest square containing a set of data (locations) """
    center = zeros([1, 2])
    radius = zeros([1, 2])

    maxX = max(location[:, 0])
    maxY = max(location[:, 1])

    minX = min(location[:, 0])
    minY = min(location[:, 1])

    center[0, 0] = 0.5 * (maxX + minX)
    center[0, 1] = 0.5 * (maxY + minY)
    radius[0, 0] = 0.5 * (maxX - minX)
    radius[0, 1] = 0.5 * (maxY - minY)

    return [center, radius]


# -------------------------------------------------------------------------------------------------------------------- #
def assign_Children(Tree, node, R, nChebNodes, cNode, TNode):
    """Assigns Children to the given node"""
    if node.N == 0:
        node.isLeaf = True
        node.isEmpty = True
    else:
        node.potential = zeros([node.N, Tree.m])
        node.nodePotential = zeros([Tree.rank, Tree.m])
        node.nodeCharge = zeros([Tree.rank, Tree.m])
        node.isEmpty = False
        node.isLeaf = False
        node.location = zeros((node.N, 2))
        # node.child = [None]*4

        get_Scaled_ChebNode(node, cNode)

        for k in range(0, node.N):
            node.location[k, :] = Tree.locationTree[node.index[k], :]

        # Operation on leaf cell 
        if node.N <= 4 * Tree.rank:
            node.isLeaf = True
            node.R = zeros((node.N, nChebNodes * nChebNodes))
            node.R = get_Transfer_From_Parent_To_Children(node.N, nChebNodes, node.location, node.center, node.radius,
                                                          TNode)
            if Tree.maxLevels < node.nLevel:
                Tree.maxLevels = node.nLevel
        else:
            for k in range(0, 4):
                node.child[k] = H2_2D_Node(node.nLevel + 1, k)
                node.child[k].parent = node

                node.child[k].center[0, 0] = node.center[0, 0] + ((k % 2) - 0.5) * node.radius[0, 0]
                node.child[k].center[0, 1] = node.center[0, 1] + ((k // 2) - 0.5) * node.radius[0, 1]
                node.child[k].radius = node.radius * 0.5
                node.child[k].N = 0

            # Assigning index number from parent to children 
            for k in range(0, node.N):
                if Tree.locationTree[node.index[k], 0] < node.center[0, 0]:
                    if Tree.locationTree[node.index[k], 1] < node.center[0, 1]:
                        # node.child[0].index = resize(node.child[0].index, [node.child[0].N + 1])
                        # node.child[0].index[node.child[0].N] = node.index[k]
                        node.child[0].index = append(node.child[0].index, node.index[k])
                        node.child[0].N += 1
                    else:
                        # node.child[2].index = resize(node.child[2].index, [node.child[2].N + 1])
                        # node.child[2].index[node.child[2].N] = node.index[k]
                        node.child[2].index = append(node.child[2].index, node.index[k])
                        node.child[2].N += 1

                else:
                    if Tree.locationTree[node.index[k], 1] < node.center[0, 1]:
                        # node.child[1].index = resize(node.child[1].index, [node.child[1].N + 1])
                        # node.child[1].index[node.child[1].N] = node.index[k]
                        node.child[1].index = append(node.child[1].index, node.index[k])
                        node.child[1].N += 1
                    else:
                        # node.child[3].index = resize(node.child[3].index, [node.child[3].N + 1])
                        # node.child[3].index[node.child[3].N] = node.index[k]
                        node.child[3].index = append(node.child[3].index, node.index[k])
                        node.child[3].N += 1

            # Calling assign children to each child 
            for k in range(0, 4):
                assign_Children(Tree, node.child[k], R, nChebNodes, cNode, TNode)
    return


# -------------------------------------------------------------------------------------------------------------------- #
def get_Scaled_ChebNode(node, cNode):
    """ Evaluates 'nChebNodes' standardized chebyshev nodes in any interval"""
    node.scaledCnode = node.center + node.radius * cNode
    return


# -------------------------------------------------------------------------------------------------------------------- #
def get_Transfer_From_Parent_To_Children(N, nChebNodes, location, center, radius, TNode):
    """Obtains interpolation operator, which interpolates information
    from Chebyshev nodes of parent to Points in children"""
    standLocation = ((location - center) / radius)
    Transfer = zeros((2, N, nChebNodes))
    for k in range(0, 2):
        # Calculating Tk (Evaluating Chebysheves at locations)
        Transfer[k, :, :] = get_Standard_Chebyshev_Polynomials(nChebNodes, N, standLocation[:, [k]])
        # Calculating Sn (Location to nodes)
        Transfer[k, :, :] = (2.0 * Transfer[k, :, :].dot(TNode.T) - 1) / nChebNodes
    R = zeros((N, nChebNodes * nChebNodes))
    for i in range(0, nChebNodes):
        for j in range(0, nChebNodes):
            R[:, i + nChebNodes * j] = Transfer[0][:, i] * Transfer[1][:, j]
    return R


# -------------------------------------------------------------------------------------------------------------------- #
def build_Tree(node):
    """Builds the FMM Tree"""
    if not node.isEmpty:
        if not node.isLeaf:
            for i in range(0, 4):
                node.child[i].neighbor[0:8] = H2_2D_Node(0, 0)
                # for j in range(0, 8):
                #    node.child[i].neighbor[j] = H2_2D_Node(0, 0)
            assign_Siblings(node)
            for k in range(0, 8):
                # if node.neighbor[k] != None:
                if (not node.neighbor[k].isLeaf) & (not node.neighbor[k].isEmpty):
                    assign_Cousin(node, k)
            for k in range(0, 4):
                build_Tree(node.child[k])
    return


# -------------------------------------------------------------------------------------------------------------------- #
def assign_Siblings(node):
    """Assign siblings to children of a the node"""

    # Assign siblings to child[0]
    node.child[0].neighbor[3] = node.child[1]
    node.child[0].neighbor[5] = node.child[2]
    node.child[0].neighbor[4] = node.child[3]

    # Assign siblings to child[1]
    node.child[1].neighbor[7] = node.child[0]
    node.child[1].neighbor[6] = node.child[2]
    node.child[1].neighbor[5] = node.child[3]

    # Assign siblings to child[2]
    node.child[2].neighbor[1] = node.child[0]
    node.child[2].neighbor[2] = node.child[1]
    node.child[2].neighbor[3] = node.child[3]

    # Assign siblings to child[3]
    node.child[3].neighbor[0] = node.child[0]
    node.child[3].neighbor[1] = node.child[1]
    node.child[3].neighbor[7] = node.child[2]

    for k in range(0, 4):
        node.child[k].nNeighbor += 3
    return


# -------------------------------------------------------------------------------------------------------------------- #
def assign_Cousin(node, neighborNumber):
    """Assign cousins to children of the node"""

    # Assigning children of neighbor 0
    if (neighborNumber == 0) & (not node.neighbor[0].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[0].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[0].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 2] = node.neighbor[0].child[2]
        node.child[0].neighbor[0] = node.neighbor[0].child[3]
        node.child[0].nInteraction += 3
        node.child[0].nNeighbor += 1

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[0].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[0].child[1]
        node.child[1].interaction[node.child[1].nInteraction + 2] = node.neighbor[0].child[2]
        node.child[1].interaction[node.child[1].nInteraction + 3] = node.neighbor[0].child[3]
        node.child[1].nInteraction += 4

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[0].child[0]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[0].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 2] = node.neighbor[0].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 3] = node.neighbor[0].child[3]
        node.child[2].nInteraction += 4

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[0].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[0].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 2] = node.neighbor[0].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 3] = node.neighbor[0].child[3]
        node.child[3].nInteraction += 4

    # Assigning children of neighbor 1
    elif (neighborNumber == 1) & (not node.neighbor[1].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[1].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[1].child[1]
        node.child[0].neighbor[1] = node.neighbor[1].child[2]
        node.child[0].neighbor[2] = node.neighbor[1].child[3]
        node.child[0].nInteraction += 2

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[1].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[1].child[1]
        node.child[1].neighbor[0] = node.neighbor[1].child[2]
        node.child[1].neighbor[1] = node.neighbor[1].child[3]
        node.child[1].nInteraction += 2

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[1].child[0]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[1].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 2] = node.neighbor[1].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 3] = node.neighbor[1].child[3]
        node.child[2].nInteraction += 4

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[1].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[1].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 2] = node.neighbor[1].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 3] = node.neighbor[1].child[3]
        node.child[3].nInteraction += 4

        # Update neighbor count.
        node.child[0].nNeighbor += 2
        node.child[1].nNeighbor += 2

    # Assigning children of neighbor 2
    elif (neighborNumber == 2) & (not node.neighbor[2].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[2].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[2].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 2] = node.neighbor[2].child[2]
        node.child[0].interaction[node.child[0].nInteraction + 3] = node.neighbor[2].child[3]
        node.child[0].nInteraction += 4

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[2].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[2].child[1]
        node.child[1].interaction[node.child[1].nInteraction + 2] = node.neighbor[2].child[3]
        node.child[1].neighbor[2] = node.neighbor[2].child[2]
        node.child[1].nInteraction += 3

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[2].child[0]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[2].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 2] = node.neighbor[2].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 3] = node.neighbor[2].child[3]
        node.child[2].nInteraction += 4

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[2].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[2].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 2] = node.neighbor[2].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 3] = node.neighbor[2].child[3]
        node.child[3].nInteraction += 4

        # Update neighbor count.
        node.child[1].nNeighbor += 1

    # Assigning children of neighbor 3
    elif (neighborNumber == 3) & (not node.neighbor[3].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[3].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[3].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 2] = node.neighbor[3].child[2]
        node.child[0].interaction[node.child[0].nInteraction + 3] = node.neighbor[3].child[3]
        node.child[0].nInteraction += 4

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].neighbor[3] = node.neighbor[3].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[3].child[1]
        node.child[1].neighbor[4] = node.neighbor[3].child[2]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[3].child[3]
        node.child[1].nInteraction += 2

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[3].child[0]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[3].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 2] = node.neighbor[3].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 3] = node.neighbor[3].child[3]
        node.child[2].nInteraction += 4

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].neighbor[2] = node.neighbor[3].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[3].child[1]
        node.child[3].neighbor[3] = node.neighbor[3].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[3].child[3]
        node.child[3].nInteraction += 2

        # Update neighbor count.
        node.child[1].nNeighbor += 2
        node.child[3].nNeighbor += 2

    # Assigning children of neighbor 4
    elif (neighborNumber == 4) & (not node.neighbor[4].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[4].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[4].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 2] = node.neighbor[4].child[2]
        node.child[0].interaction[node.child[0].nInteraction + 3] = node.neighbor[4].child[3]
        node.child[0].nInteraction += 4

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[4].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[4].child[1]
        node.child[1].interaction[node.child[1].nInteraction + 2] = node.neighbor[4].child[2]
        node.child[1].interaction[node.child[1].nInteraction + 3] = node.neighbor[4].child[3]
        node.child[1].nInteraction += 4

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[4].child[0]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[4].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 2] = node.neighbor[4].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 3] = node.neighbor[4].child[3]
        node.child[2].nInteraction += 4

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].neighbor[4] = node.neighbor[4].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[4].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[4].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 2] = node.neighbor[4].child[3]
        node.child[3].nInteraction += 3

        # Update neighbor count.
        node.child[3].nNeighbor += 1

    # Assigning children of neighbor 5
    elif (neighborNumber == 5) & (not node.neighbor[5].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[5].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[5].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 2] = node.neighbor[5].child[2]
        node.child[0].interaction[node.child[0].nInteraction + 3] = node.neighbor[5].child[3]
        node.child[0].nInteraction += 4

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[5].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[5].child[1]
        node.child[1].interaction[node.child[1].nInteraction + 2] = node.neighbor[5].child[2]
        node.child[1].interaction[node.child[1].nInteraction + 3] = node.neighbor[5].child[3]
        node.child[1].nInteraction += 4

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].neighbor[5] = node.neighbor[5].child[0]
        node.child[2].neighbor[4] = node.neighbor[5].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[5].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[5].child[3]
        node.child[2].nInteraction += 2

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].neighbor[6] = node.neighbor[5].child[0]
        node.child[3].neighbor[5] = node.neighbor[5].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[5].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[5].child[3]
        node.child[3].nInteraction += 2

        # Update neighbor count.
        node.child[2].nNeighbor += 2
        node.child[3].nNeighbor += 2

    # Assigning children of neighbor 6
    elif (neighborNumber == 6) & (not node.neighbor[6].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[6].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[6].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 2] = node.neighbor[6].child[2]
        node.child[0].interaction[node.child[0].nInteraction + 3] = node.neighbor[6].child[3]
        node.child[0].nInteraction += 4

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[6].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[6].child[1]
        node.child[1].interaction[node.child[1].nInteraction + 2] = node.neighbor[6].child[2]
        node.child[1].interaction[node.child[1].nInteraction + 3] = node.neighbor[6].child[3]
        node.child[1].nInteraction += 4

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[6].child[0]
        node.child[2].neighbor[6] = node.neighbor[6].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[6].child[2]
        node.child[2].interaction[node.child[2].nInteraction + 2] = node.neighbor[6].child[3]
        node.child[2].nInteraction += 3

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[6].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[6].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 2] = node.neighbor[6].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 3] = node.neighbor[6].child[3]
        node.child[3].nInteraction += 4

        # Update neighbor count.
        node.child[2].nNeighbor += 1

    # Assigning children of neighbor 7
    elif (neighborNumber == 7) & (not node.neighbor[7].isEmpty):
        # Assigning the cousins to child0. One neighbor and three well-separated cousins.
        node.child[0].interaction[node.child[0].nInteraction + 0] = node.neighbor[7].child[0]
        node.child[0].neighbor[7] = node.neighbor[7].child[1]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[7].child[2]
        node.child[0].neighbor[6] = node.neighbor[7].child[3]
        node.child[0].nInteraction += 2

        # Assigning the cousins to child1. Four well-separated cousins.
        node.child[1].interaction[node.child[1].nInteraction + 0] = node.neighbor[7].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[7].child[1]
        node.child[1].interaction[node.child[1].nInteraction + 2] = node.neighbor[7].child[2]
        node.child[1].interaction[node.child[1].nInteraction + 3] = node.neighbor[7].child[3]
        node.child[1].nInteraction += 4

        # Assigning the cousins to child2. Four well-separated cousins.
        node.child[2].interaction[node.child[2].nInteraction + 0] = node.neighbor[7].child[0]
        node.child[2].neighbor[0] = node.neighbor[7].child[1]
        node.child[2].interaction[node.child[2].nInteraction + 1] = node.neighbor[7].child[2]
        node.child[2].neighbor[7] = node.neighbor[7].child[3]
        node.child[2].nInteraction += 2

        # Assigning the cousins to child3. Four well-separated cousins.
        node.child[3].interaction[node.child[3].nInteraction + 0] = node.neighbor[7].child[0]
        node.child[3].interaction[node.child[3].nInteraction + 1] = node.neighbor[7].child[1]
        node.child[3].interaction[node.child[3].nInteraction + 2] = node.neighbor[7].child[2]
        node.child[3].interaction[node.child[3].nInteraction + 3] = node.neighbor[7].child[3]
        node.child[3].nInteraction += 4

        # Update neighbor count.
        node.child[0].nNeighbor += 2
        node.child[2].nNeighbor += 2
    return

########################################################################################################################
