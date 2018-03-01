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
from time import perf_counter
from numpy import loadtxt, dot
from numpy.linalg import norm
from CustomKernels import CustomKernels
from H2_2D_Tree import H2_2D_Tree
from kernel_Base import calculate_Potential

# GET INPUT DATA FROM DATA FILE -------------------------------------------------------------------------------------- #
# Read data from input file
Data = loadtxt('Input//input.txt')

location = Data[0:10000, 0:2]   # Locations of the charges matrix
charges = Data[0:10000, 2:]     # Sets of charges
N = len(location)               # Number of points
m = charges.shape[1]            # Sets of charge vectors
nChebNodes = 5                  # Number of Chebyshev nodes( >= 3)

print(' Number of charges: %d ' % N)
print(' Number of sets of charges: %d ' % m)
print(' Number of Chebyshev Nodes: %d ' % nChebNodes)

# FAST MATRIX VECTOR PRODUCT ----------------------------------------------------------------------------------------- #
# 1 Building FMM Tree
start = perf_counter()
ATree = H2_2D_Tree(nChebNodes, charges, location, N, m)  # FMM Tree
print(' Total time taken for FMM(build tree) is: %f  seconds' % (perf_counter() - start))

# 2 Calculating Potential
start = perf_counter()
kex1 = CustomKernels.exampleKernelA  # Name of the custom kernel
potential_kex1 = calculate_Potential(kex1, ATree, charges)
print(' Total time taken for FMM(calculations) is: %f seconds' % (perf_counter() - start))

# EXACT MATRIX VECTOR PRODUCT ---------------------------------------------------------------------------------------- #
# Calculate potential Exact
start = perf_counter()
print('\n Starting exact computation...')
Q = kex1(N, location, N, location)
potential_exact = dot(Q, charges)
print(' Done.')
print(' Total time taken for Exact(calculations) is: %f seconds' % (perf_counter() - start))
print('\n Maximum Error is: %.3e \n' % (norm(potential_exact - potential_kex1) / norm(potential_exact)))

########################################################################################################################
