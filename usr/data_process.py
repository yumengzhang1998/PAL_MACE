import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation 
import numpy as np
import ast
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
import matplotlib.pyplot as plt
import re
# Path: ALParallel/usr/data_process.py
# Data Path: ALParallel/results/TestRun/added_data.csv
radius_graph_transform = T.RadiusGraph(r=4.0)  
def calculate_normal_vector(points):
    """Calculate the normal vector of a plane defined by three points."""
    # Calculate two vectors from the points
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    # Cross product of the two vectors
    normal = np.cross(v1, v2)
    # Normalize the normal vector
    normal /= np.linalg.norm(normal)
    return normal

def calculate_dihedral_angle(plane1_points, plane2_points):
    """Calculate the dihedral angle between two planes defined by three points each."""
    # Calculate normal vectors for each plane
    normal1 = calculate_normal_vector(plane1_points)
    normal2 = calculate_normal_vector(plane2_points)
    # Calculate the angle between the normals
    dot_product = np.dot(normal1, normal2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def clear_coord(coord):

                
    return np.fromstring(coord.strip("[]").replace("\n", " ").replace("[", "").replace("]", ""), sep=" ").reshape(-1, 3)

def extract_int_from_tensor_string(tensor_string):
    # Use regular expression to find the integer part of the string
    match = re.search(r'tensor\((-?\d+)', tensor_string)
    if match:
        # Convert the matched string to an integer
        return int(match.group(1))
    else:
        raise ValueError("The input string does not contain a valid tensor representation.")

def row_to_data(row):
    x = torch.tensor(row['node_feature'], dtype=torch.float)
    
    # Create a Data object
    data = Data(x=x, pos = x)
    
    # Add any additional attributes as necessary
    data.global_charge = torch.tensor([row['global_charge']], dtype=torch.float)
    data.y = torch.tensor([row['energy']], dtype=torch.float)
    data.type = row['type']
    
    return data


def convert_to_numpy_array(value):
    try:
        # Replace the 'tensor' part and the dtype specification to make it a valid list
        value = value.replace('tensor(', '').replace('dtype=torch.float64', '').replace(')', '').strip()
        # Use ast.literal_eval to safely evaluate the string to a list
        value = ast.literal_eval(value)
        # Convert the list to a NumPy array
        return np.array(value)
    except (ValueError, SyntaxError):
        return value




df = pd.read_csv('../results/TestRun/added_data.csv')
train_df = df[df['type'] == 'train'].iloc[449:]

# Filter rows for 'val' and select rows from the 50th onward (0-indexed, so 49)
val_df = df[df['type'] == 'val'].iloc[49:]
# Combine the filtered dataframes back together if needed
df = pd.concat([train_df, val_df]).reset_index(drop=True)
print(len(df))
# quit()
df.to_csv('../results/TestRun/generated_data.csv', index=False)

a = df['node_feature'][0]
numpy_array = convert_to_numpy_array(a)

df['node_feature'] = df['node_feature'].apply(lambda x: convert_to_numpy_array(x)) 

df['node_feature'] = [x.reshape(4, 3) for x in df['node_feature']]

df['energy'] = df['energy'].apply(lambda x: convert_to_numpy_array(x)) 
print(df['energy'])

df['global_charge'] = [extract_int_from_tensor_string(s) for s in df['global_charge']]

data_list = [row_to_data(row) for _, row in df.iterrows()]
trasnformed = [radius_graph_transform(x) for x in data_list]

edge_indices = [i.edge_index for i in trasnformed]


num_frames = len(df)
positions =  np.array(df['node_feature'].to_list()) # Random 3D positions for 4 points in each frame
df['dihedral'] = [calculate_dihedral_angle([mol[0], mol[1], mol[2]], [mol[3], mol[1], mol[2]]) for mol in positions]
df.to_csv('../results/TestRun/generated_data_with_angle.csv', index=False)
quit()
print('make figure')

plt.scatter(df['dihedral'], df['energy'])

# Add title and labels
plt.title('Dihedral Angle vs Energy')
plt.xlabel('Diheral Angle')
plt.ylabel('Energy')

# Show plot
plt.savefig('dihedral_energy.png')





x_max = np.max(positions[:, :, 0])
x_min = np.min(positions[:, :, 0])
y_max = np.max(positions[:, :, 1])
y_min = np.min(positions[:, :, 1])
z_max = np.max(positions[:, :, 2])
z_min = np.min(positions[:, :, 2])







fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([], [], [])
lines = []

def init():
    return scatter, *lines

def update(frame):
    # Clear previous lines
    for line in lines:
        line.remove()
    lines.clear()

    # Update scatter plot with current frame's positions
    scatter._offsets3d = (positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2])

    # Get edge index for the current frame
    edge_index = edge_indices[frame]

    # Plot edges based on edge_index tensor
    for i in range(edge_index.size(1)):
        start_idx = edge_index[0, i]
        end_idx = edge_index[1, i]
        line = ax.plot(
            [positions[frame, start_idx, 0], positions[frame, end_idx, 0]],
            [positions[frame, start_idx, 1], positions[frame, end_idx, 1]],
            [positions[frame, start_idx, 2], positions[frame, end_idx, 2]],
            color='blue', linestyle='--'
        )[0]
        lines.append(line)

    ax.set_title('Molecule {}'.format(frame))
    return scatter, *lines

# Number of frames for the animation
num_frames = len(edge_indices)
print('make animation')
# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=100, init_func=init)  # Adjust interval as needed

# Save animation
writervideo = animation.FFMpegWriter(fps=60) 
ani.save('increasingStraightLine.mp4', writer=writervideo) 

plt.show()





# Create figure and 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter([], [], [])
# lines = []
# def init():
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_zlabel('Z')
#     # ax.set_title('Frame')
#     # ax.set_xlim([x_min, x_max])
#     # ax.set_ylim([y_min, y_max])
#     # ax.set_zlim([z_min, z_max])

#     return scatter,*lines


# # Function to update the plot for each frame
# def update(frame):
#     # Clear the previous frame's points
#     # ax.cla()
    
#     # Plot the points for the current frame
#     # ax.scatter(positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2])
    
#     # Set axis labels and title
#     ax.set_title('Molecule {}'.format(frame))
    

#     scatter._offsets3d = (positions[frame, :, 0], positions[frame, :, 1], positions[frame, :, 2])
#     for line in lines:
#         line.remove()
#     lines.clear()
#     # for i in range(len(positions[frame])):
#     #     line = ax.plot(
#     #         [positions[frame, i, 0], positions[frame, (i+1)%4, 0]],
#     #         [positions[frame, i, 1], positions[frame, (i+1)%4, 1]],
#     #         [positions[frame, i, 2], positions[frame, (i+1)%4, 2]],
#     #         color='blue', linestyle='--'
#     #     )[0]
#     #     lines.append(line)

#     lines.append(ax.plot(
#         [positions[frame, 0, 0], positions[frame, 1, 0]],
#         [positions[frame, 0, 1], positions[frame, 1, 1]],
#         [positions[frame, 0, 2], positions[frame, 1, 2]],
#         color='blue', linestyle='--'
#     )[0])
#     lines.append(ax.plot(
#         [positions[frame, 0, 0], positions[frame, 2, 0]],
#         [positions[frame, 0, 1], positions[frame, 2, 1]],
#         [positions[frame, 0, 2], positions[frame, 2, 2]],
#         color='blue', linestyle='--'
#     )[0])
#     lines.append(ax.plot(
#         [positions[frame, 3, 0], positions[frame, 1, 0]],
#         [positions[frame, 3, 1], positions[frame, 1, 1]],
#         [positions[frame, 3, 2], positions[frame, 1, 2]],
#         color='blue', linestyle='--'
#     )[0])
#     lines.append(ax.plot(
#         [positions[frame, 3, 0], positions[frame, 2, 0]],
#         [positions[frame, 3, 1], positions[frame, 2, 1]],
#         [positions[frame, 3, 2], positions[frame, 2, 2]],
#         color='blue', linestyle='--'
#     )[0])

#     # for i, pos in enumerate(positions[frame]):
#     #     ax.text(pos[0], pos[1], pos[2], f'Atom {i+1}', color='black', fontsize=8)
#     return scatter,
# # Create animation
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100, init_func=init)  # Adjust interval as needed

# writervideo = animation.FFMpegWriter(fps=60) 
# ani.save('increasingStraightLine.mp4', writer=writervideo) 
# # Show plot
# plt.show()