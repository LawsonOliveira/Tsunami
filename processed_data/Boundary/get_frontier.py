import numpy as np
import pyvista as pv


# Reading files
mesh = pv.read("./mesh_arcachon.vtk")
ncells = mesh.GetNumberOfCells()
data_cells = np.array([np.array(mesh.GetCell(i).GetPoints().GetData()) for i in range(mesh.GetNumberOfCells())])



def get_neighbors_points_for_each_cell(grid, cell_idx):
    ### Helper to get neighbor points for each cell
    ### a neighbor point is a point inside a neighbor cell
    cell = grid.GetCell(cell_idx)
    pids = pv.vtk_id_list_to_array(cell.GetPointIds())
    neighbors = set(grid.extract_points(pids)["vtkOriginalCellIds"])
    neighbors.discard(cell_idx)
    neighbors = np.array(list(neighbors))
    neighs_cells = grid.extract_cells(neighbors)
    neighs_points = np.array(neighs_cells.GetPoints().GetData())
    return neighs_points
    


### Get neighbor points for each cell
print(f"First stage")
neighbors_points = []
for i in range(ncells):
    if i%1000==999:
        print(f'{i+1} out of {ncells}')
    neighbors_points.append(get_neighbors_points_for_each_cell(mesh, i))

np.save("./neighbors_points.npy", neighbors_points, allow_pickle=True)



### Get cells that in the boundary
print(f"Second stage")
frontier_cells = []
for i in range(ncells):
    if i%1000==999:
        print(f'{i+1} out of {ncells}')
    if (neighbors_points[i][:,2]>0).any() and (neighbors_points[i][:,2]<0).any():
        frontier_cells.append(i)

np.save("./frontier_cells.npy", frontier_cells, allow_pickle=True)



### Get points that in the boundary
print(f"Third stage")
points = []
count = 0
for cell in frontier_cells:
    if count%1000==999:
        print(f'{count+1} out of {len(frontier_cells)}')
    for i in range(data_cells[cell].shape[0]):
        points.append(data_cells[cell][i])
    count += 1

points = np.array(points)
points = np.unique(points, axis=0)
print(f"Number of cell in the frontier: {len(frontier_cells)}")
print(f"Number of points in the frontier: {points.shape[0]}")
np.savetxt("./frontier.csv", points, delimiter=",")

