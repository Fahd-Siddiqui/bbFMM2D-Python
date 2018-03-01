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
from numpy import zeros, dot, array, meshgrid


# -------------------------------------------------------------------------------------------------------------------- #
def calculate_Potential(custom_Kernel, tree, charges):
    """Calculates the potential for the node and transfers it to the Tree"""
    potential = zeros([tree.N, tree.m])
    set_Tree_Potential_Zero(tree.root, tree.rank)
    set_Node_Charge_Zero(tree.root, tree.rank)
    tree.chargeTree = charges
    update_Charge(tree, tree.root)
    print('\n Calculating potential...')
    potential = calculate_Potential_4Args(custom_Kernel, tree, tree.root, potential)
    print(' Done.')
    return potential


# -------------------------------------------------------------------------------------------------------------------- #
def set_Tree_Potential_Zero(node, rank):
    """Sets tree potential to zero"""
    if not node.isEmpty:
        node.potential = zeros([node.N, 1])
        node.nodePotential = zeros([rank, 1])
        for k in range(0, 4):
            set_Tree_Potential_Zero(node.child[k], rank)
    return


# -------------------------------------------------------------------------------------------------------------------- #
def set_Node_Charge_Zero(node, rank):
    """Sets node charge to zero"""
    if not node.isEmpty:
        node.chargeComputed = False
        node.charge = zeros([node.N, 1])
        node.nodeCharge = zeros([rank, 1])
        for k in range(0, 4):
            set_Node_Charge_Zero(node.child[k], rank)
    return


# -------------------------------------------------------------------------------------------------------------------- #
def update_Charge(Tree, node):
    """Updates the charges on the Tree"""
    # Operation on leaf cell ---------------------------------------------------
    if node.isLeaf:
        get_Charge(Tree, node)

        #  Step one from the paper (page 5 Fong et al 2009)
        node.nodeCharge += dot(node.R.T, node.charge)
    else:
        # Calling assign children to each child --------------------------------
        for k in range(0, 4):
            update_Charge(Tree, node.child[k])
            if not node.child[k].isEmpty:
                # Step two from the paper (page 5 Fong et al 2009)
                node.nodeCharge += dot((Tree.R[k, :, :]).T, node.child[k].nodeCharge)
    return


# -------------------------------------------------------------------------------------------------------------------- #
def get_Charge(Tree, node):
    """Obtains charge to node when needed"""
    if node.chargeComputed:
        return
    else:
        node.chargeComputed = True
        node.charge = Tree.chargeTree[node.index]
    return


# -------------------------------------------------------------------------------------------------------------------- #
def calculate_Potential_4Args(custom_Kernel, tree, node, potential):
    if not node.isEmpty:
        if node.isLeaf:
            if not node.isRoot:
                for k in range(0, 8):
                    if not node.neighbor[k].isEmpty:
                        # Potential from neighbors
                        get_Charge(tree, node.neighbor[k])
                        node.potential += dot(
                            custom_Kernel(node.N, node.location, node.neighbor[k].N, node.neighbor[k].location),
                            node.neighbor[k].charge)

            # Potential from Chebyshev nodes (Local expansion)
            node.potential += dot(node.R, node.nodePotential)

            # Self potential
            node.potential += dot(custom_Kernel(node.N, node.location, node.N, node.location), node.charge)
            potential = tranfer_Potential_To_Potential_Tree(node, potential)
        else:
            computePotential = False
            for k in range(0, 8):
                if not node.isRoot:
                    if not node.neighbor[k].isEmpty:
                        if node.neighbor[k].isLeaf:
                            get_Charge(tree, node.neighbor[k])
                            node.potential += \
                                dot(custom_Kernel(node.N, node.location, node.neighbor[k].N, node.neighbor[k].location),
                                    node.neighbor[k].charge)
                            computePotential = True

            # M2L Step three from the paper (page 5 Fong et al 2009)
            calculate_NodePotential_From_Wellseparated_Clusters(custom_Kernel, node, tree.nChebNodes)

            # L2L Step four from the paper (page 5 Fong et al 2009)
            transfer_NodePotential_To_Child(node, tree.R)

            if computePotential:
                potential = tranfer_Potential_To_Potential_Tree(node, potential)

            for k in range(0, 4):
                potential = calculate_Potential_4Args(custom_Kernel, tree, node.child[k], potential)

    return potential


# -------------------------------------------------------------------------------------------------------------------- #
def tranfer_Potential_To_Potential_Tree(node, potential):
    """Tranfers potential from node to final potential matrix when needed"""
    potential[node.index, 0] = potential[node.index, 0] + node.potential[0: node.N, 0]
    return potential


# -------------------------------------------------------------------------------------------------------------------- #
def calculate_NodePotential_From_Wellseparated_Clusters(custom_Kernel, node, nChebNodes):
    """M2L Obtains Chebyshev node potential from well separated clusters"""
    for k in range(0, 4):
        if not node.child[k].isEmpty:
            for i in range(0, node.child[k].nInteraction):
                if not node.child[k].interaction[i].isEmpty:
                    K = kernel_Cheb_2D(custom_Kernel, nChebNodes, node.child[k].scaledCnode, nChebNodes,
                                       node.child[k].interaction[i].scaledCnode)
                    # M2L Step three from the paper (page 5 Fong et al 2009)
                    node.child[k].nodePotential += dot(K, node.child[k].interaction[i].nodeCharge)
    return


# -------------------------------------------------------------------------------------------------------------------- #
def kernel_Cheb_2D(custom_Kernel, M, xVec, N, yVec):
    """Evaluate kernel at Chebyshev nodes"""
    # from numpy import append, delete, empty
    # mbym = empty((1, 2))
    # nbyn = empty((1, 2))
    # for j in range(0, M):
    #     for i in range(0, M):
    #         mbym = append(mbym, [[xVec[i, 0], xVec[j, 1]]], axis=0)
    #
    # for j in range(0, N):
    #     for i in range(0, N):
    #         nbyn = append(nbyn, [[yVec[i, 0], yVec[j, 1]]], axis=0)
    #
    # mbym = delete(mbym, 0, axis=0)
    # nbyn = delete(nbyn, 0, axis=0)

    mbym = zeros((M * M, 2))
    nbyn = zeros((N * N, 2))

    for i in range(0, M * M, M):
        mbym[i: i + M, 0] = xVec[0: M, 0]
        mbym[i: i + M, 1] = xVec[i // M, 1]

    for i in range(0, N * N, N):
        nbyn[i: i + N, 0] = yVec[0: N, 0]
        nbyn[i: i + N, 1] = yVec[i // N, 1]

    # mbym=array(meshgrid(xVec[:,0], xVec[:,1])).reshape(2, -1).T
    # nbyn=array(meshgrid(yVec[:,0], yVec[:,1])).reshape(2, -1).T

    K = custom_Kernel(M * M, mbym, N * N, nbyn)
    return K


# -------------------------------------------------------------------------------------------------------------------- #
def transfer_NodePotential_To_Child(node, R):
    """L2L Tranfers potential from Chebyshev node of parent to Chebyshev node of children"""

    # L2L Step four from the paper (page 5 Fong et al 2009)
    for k in range(0, 4):
        if not node.child[k].isEmpty:
            node.child[k].nodePotential += dot(R[k, :, :], node.nodePotential)
    return

########################################################################################################################
