"""
Evaluate the TSRS data
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import yaml 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--case_id", type=int, default=302000)
args = parser.parse_args()

def plot_data(fig, ax, data_dict, ID=0, var='u'):
    """
    Plot a single variable on a single axis
    """
    plt.rc("font",family = "serif")
    plt.rc("font",size = 12)
    plt.rc("axes",labelsize =12, linewidth = 1)
    plt.rc("legend",fontsize= 12, handletextpad = 0.1)
    plt.rc("xtick",labelsize = 12)
    plt.rc("ytick",labelsize = 12)
    
    Nx, Nz = data_dict[var].shape[0], data_dict[var].shape[1]
    x = np.linspace(0, 2*np.pi, Nx)
    z = np.linspace(0, np.pi, Nz)
    X, Z = np.meshgrid(x, z)
    # Calculate the fluctuation
    data_fluc = data_dict[var].copy()
    data_fluc -= np.mean(data_fluc, axis=-1, keepdims=True)
    contour = ax.contourf(
        X, Z, data_fluc[:,:,ID].T, levels=100, cmap='turbo')
    fig.colorbar(contour, ax=ax, **{'orientation':'vertical','pad':0.02,'shrink':1})
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(var)
    return fig, ax, {'x': X, 'z': Z, var: data_dict[var].copy()}


if __name__ == "__main__":

    Head = "TCF_1" ;case_ids = [302000,402032,302999]
    Head = "TCF_2" ; case_ids = [301000,401031,301999,]
    #Head = "TCF_3" ; case_ids = [304000,403033,304999,]
    
    # Collect all data paths for y15 and wall
    all_y15_paths = []
    all_wall_paths = []
    case_y15_mapping = []  # Track which case each path belongs to
    case_wall_mapping = []
    
    for case_id in case_ids:
        print(f"Processing case {case_id}")
        print("--------------------------------")
        config_path = os.path.join(f"../runs/{case_id}/current_conf.yml") 
        data_path = os.path.join(f"../runs/{case_id}")
        path_list = os.listdir(data_path)
        path_list = [os.path.join(data_path,f) for f in path_list if "env" in f]
        path_list.sort()
        y15_path = []
        wall_path = []
        for path in path_list:
            path_list = os.listdir(path)
            path_list = [os.path.join(path,f) for f in path_list if "tsrs" in f]
            path_list.sort()
            if len(path_list) > 0:
                y15_path += [f for f in path_list if "y15" in f]
                wall_path += [f for f in path_list if "wall" in f]
            else:
                print(f"No data found in {path}")
        
        # Use first file from each case if multiple exist
        if len(y15_path) > 0:
            all_y15_paths.append(y15_path[0])
            case_y15_mapping.append(case_id)
        if len(wall_path) > 0:
            all_wall_paths.append(wall_path[0])
            case_wall_mapping.append(case_id)

    print("--------------------------------")
    
    # Plot y15 data: one figure with all cases
    if len(all_y15_paths) > 0:
        num_cases = len(all_y15_paths)
        vars = ['u', 'v', 'w']
        fig, axs = plt.subplots(num_cases, 3, figsize=(12, 4*num_cases), sharex=True, sharey=True)
        
        # Handle single case (axs will be 1D instead of 2D)
        if num_cases == 1:
            axs = axs.reshape(1, -1)
        
        for i, (path, case_id) in enumerate(zip(all_y15_paths, case_y15_mapping)):
            print(f"Plotting y15 data for case {case_id}")
            data_dict = sio.loadmat(path)
            gen_data = {}
            for j, var in enumerate(vars):
                fig, axs[i, j], var_data = plot_data(fig, axs[i, j], data_dict, ID=200, var=var)
                gen_data[var] = var_data[var]
                if 'x' not in gen_data:
                    gen_data['x'] = var_data['x']
                    gen_data['z'] = var_data['z']
            
            # Set labels only on appropriate axes
            axs[i, 1].set_ylabel(f'Case {case_id}\nZ')
            if i == num_cases - 1:  # Last row
                axs[i, 0].set_xlabel('X')
                axs[i, 2].set_xlabel('X')
            
            sio.savemat(os.path.join(f"datas/tsrs_y15_{case_id}.mat"), gen_data)
        
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.savefig(os.path.join(f"Figs/{Head}_tsrs_y15_all_cases.png"), bbox_inches='tight')
        plt.close(fig)

    # Plot wall data: one figure with all cases
    if len(all_wall_paths) > 0:
        num_cases = len(all_wall_paths)
        vars = ['u', 'v', 'w']
        fig, axs = plt.subplots(num_cases, 3, figsize=(12, 4*num_cases), sharex=True, sharey=True)
        
        # Handle single case (axs will be 1D instead of 2D)
        if num_cases == 1:
            axs = axs.reshape(1, -1)
        
        for i, (path, case_id) in enumerate(zip(all_wall_paths, case_wall_mapping)):
            print(f"Plotting wall data for case {case_id}")
            data_dict = sio.loadmat(path)
            gen_data = {}
            for j, var in enumerate(vars):
                fig, axs[i, j], var_data = plot_data(fig, axs[i, j], data_dict, ID=200, var=var)
                gen_data[var] = var_data[var]
                if 'x' not in gen_data:
                    gen_data['x'] = var_data['x']
                    gen_data['z'] = var_data['z']
            
            # Set labels only on appropriate axes
            axs[i, 1].set_ylabel(f'Case {case_id}\nZ')
            if i == num_cases - 1:  # Last row
                axs[i, 0].set_xlabel('X')
                axs[i, 2].set_xlabel('X')
            
            sio.savemat(os.path.join(f"datas/tsrs_wall_{case_id}.mat"), gen_data)
        
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.savefig(os.path.join(f"Figs/{Head}_tsrs_wall_all_cases.png"), bbox_inches='tight')
        plt.close(fig)
