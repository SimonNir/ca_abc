import numpy as np
from ca_abc.potentials import StandardMullerBrown2D

# Initialize potential
pot = StandardMullerBrown2D()

# Define known minima
A = np.array([-0.5582236346, 1.441725842])        # Basin A
B = np.array([0.6234994049, 0.02803775853])       # Basin B  
C = np.array([-0.050010823, 0.4666941049])        # Basin C

# Define known saddles
saddle_BC = np.array([0.212486582, 0.2929883251])   # Saddle between B and C
saddle_CA = np.array([-0.8220015587, 0.6243128028]) # Saddle between C and A

# Step 1: Average nearest saddle distance
minima = [A, B, C]
saddles = [saddle_BC, saddle_CA]

nearest_distances = []
for minimum in minima:
    distances = [np.linalg.norm(minimum - saddle) for saddle in saddles]
    nearest_distances.append(min(distances))

average_distance = np.mean(nearest_distances)

# Step 2: Compute potential energies at minima and saddles
V_A = pot.potential(A)
V_B = pot.potential(B)
V_C = pot.potential(C)
V_saddle_BC = pot.potential(saddle_BC)
V_saddle_CA = pot.potential(saddle_CA)

# Step 3: Barrier heights for B → C → A path
barrier_B_to_C = V_saddle_BC - V_B
barrier_C_to_A = V_saddle_CA - V_C
barriers_BCA = np.array([barrier_B_to_C, barrier_C_to_A])
average_barrier_BCA = np.mean(barriers_BCA)
std_barrier_BCA = np.std(barriers_BCA)

# Step 4: Barrier heights for A → C → B path
barrier_A_to_C = V_saddle_CA - V_A
barrier_C_to_B = V_saddle_BC - V_C
barriers_ACB = np.array([barrier_A_to_C, barrier_C_to_B])
average_barrier_ACB = np.mean(barriers_ACB)
std_barrier_ACB = np.std(barriers_ACB)

# Output results
print(f"Average nearest saddle distance: {average_distance:.6f}\n")

print("Barriers along B → C → A path:")
print(f"  Barrier B → C: {barrier_B_to_C:.6f}")
print(f"  Barrier C → A: {barrier_C_to_A:.6f}")
print(f"  Average barrier: {average_barrier_BCA:.6f} ± {std_barrier_BCA:.6f}\n")

print("Barriers along A → C → B path:")
print(f"  Barrier A → C: {barrier_A_to_C:.6f}")
print(f"  Barrier C → B: {barrier_C_to_B:.6f}")
print(f"  Average barrier: {average_barrier_ACB:.6f} ± {std_barrier_ACB:.6f}")
