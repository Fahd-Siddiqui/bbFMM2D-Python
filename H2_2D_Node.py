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
from numpy import zeros, array


# -------------------------------------------------------------------------------------------------------------------- #
class H2_2D_Node:
    """H2_2D_Node Class for the nodes of the FMM Tree"""
    isEmpty = True
    isLeaf = True

    def __init__(self, nLevel, nodeNumber):
        """Constructor initializes each node of the FMM Tree"""
        self.neighbor = array([H2_2D_Node for count in range(8)])
        self.interaction = array([H2_2D_Node for count in range(27)])
        self.center = zeros((1, 2))
        self.charge = 0
        self.chargeComputed = False
        self.child = array([H2_2D_Node for count in range(4)])
        self.index = array([], dtype=int)
        self.isEmpty = True
        self.isLeaf = True
        self.isRoot = False
        self.location = zeros((1, 2))
        self.N = 0
        self.nInteraction = 0
        self.nLevel = nLevel
        self.nNeighbor = 0
        self.nodeCharge = zeros([])
        self.nodeNumber = nodeNumber
        self.nodePotential = zeros([])
        self.parent = H2_2D_Node
        self.potential = zeros((1, 1))
        self.R = zeros([])
        self.radius = zeros((1, 2))
        self.scaledCnode = zeros((1, 2))

########################################################################################################################
