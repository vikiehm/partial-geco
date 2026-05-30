from utils.partial_geco import solve_partial_geco
from utils.coarse_to_fine import coarse_to_fine

if __name__ == "__main__":
    map_filename = "dog-15_dog-6"
    lamda = 0.3
    # First solve the partial GECO optimization
    solve_partial_geco(map_filename=map_filename, lamda=lamda)

    # Then perform coarse-to-fine optimization with slack variables
    coarse_to_fine(old_nr=600, new_nr=800, map_filename=map_filename, lamda=lamda)
    coarse_to_fine(old_nr=800, new_nr=1000, map_filename=map_filename, lamda=lamda)
