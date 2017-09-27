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

from H2_2D_Tree_Functions import *
from numpy import arange


# -------------------------------------------------------------------------------------------------------------------- #

class H2_2D_Tree:
    """H2_2D_TREE Class object for the FMM Tree"""

    def __init__(self, nChebNodes, charge, location, N, m):
        """Constructor initializes the FMM Tree"""
        self.nChebNodes = nChebNodes
        self.rank = nChebNodes * nChebNodes
        self.N = N
        self.m = m
        self.maxLevels = 0
        self.chargeTree = charge
        self.locationTree = location

        # Get Chebyshev nodes
        self.cNode = get_Standard_Chebyshev_Nodes(self.nChebNodes)

        # Get Chebyshev polynomials evaluated at Chebyshev nodes
        self.TNode = get_Standard_Chebyshev_Polynomials(self.nChebNodes, self.nChebNodes, self.cNode)

        # Gets transfer matrices
        self.R = get_Transfer(self.nChebNodes, self.cNode, self.TNode)

        (self.center, self.radius) = get_Center_Radius(location)

        # Create root
        self.root = H2_2D_Node(0, 0)
        self.root.nNeighbor = 0
        self.root.nInteraction = 0
        self.root.N = N
        self.root.center = self.center
        self.root.radius = self.radius
        self.root.index = arange(0, N)
        self.root.isRoot = True

        print('\n Assigning children...')
        assign_Children(self, self.root, self.R, nChebNodes, self.cNode, self.TNode)
        print(' Done.')
        build_Tree(self.root)
        print('\n Maximum levels is: %d' % self.maxLevels)

########################################################################################################################
