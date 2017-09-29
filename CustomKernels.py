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

from numpy import zeros, indices, sqrt
import math


# import other modules as necessary

# -------------------------------------------------------------------------------------------------------------------- #
class CustomKernels:
    """Class for defining the custom kernel functions"""

    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def exampleKernelA(M, x, N, y):
        """Example kernel function A"""
		# Euclidean norm function implemented using for loops. DO NOT USE EXTREMELY SLOW.
        kernel = zeros([M, N])
        for i in range(0, M):
            for j in range(0, N):
                # Define the custom kernel function here
                kernel[i, j] = math.sqrt((x[i, 0] - y[j, 0]) ** 2 + (x[i, 1] - y[j, 1]) ** 2)
        return kernel

    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def exampleKernelB(M, x, N, y):
        """Example kernel function B"""
		# Euclidean norm function implemented using meshgrid idea.
		# Python way of defining it and much(100 times) faster than using for loops
        i, j=indices((M,N))
        kernel = zeros((M, N))
		# Define custom kernel here
        kernel[i, j] = sqrt((x[i, 0] - y[j, 0]) ** 2 + (x[i, 1] - y[j, 1]) ** 2)
        return kernel

# Define more custom kernel functions here

########################################################################################################################
