"""
Collection of NEK5000 usage 
"""
import os 
import subprocess,shutil
from configs_meta import Simulation as nek
from configs_meta import Runner as drl
from pathlib import Path
from configs_meta import param_mapping
"""
A checklist of dependencies: 

        - CASE.re2
        - CASE.ma2
        - CASE.par
        - SESSION.NAME
        - int_pos (if needed)
        - nek5000 

"""

class NEK_INIT():
    def __init__(self,nek:nek,drl:drl,rank_folder) -> None:
        """
        A class for initialization of NEK Dependencies 
        nek:[dataclass]Simulation config
        drl:[dataclass]DRL config
        rank_folder:[str]target folders to run drl 
        """
        self.nek = nek 
        self.drl = drl 
        self.rank_folder = rank_folder
        self.is_done=[]
    #-----------------------------------------
    def get_Case_Files(self):
        """
        Get required case files for running simulation
        IF it is complusory, it will be rewritten no matter if the file exists 
        IF it is optional, it will NOT be covered if it Exist.
        """
        checklist = {
                    'must':[
                            # Solver
                            "nek5000",
                            # Mesh 
                            f"{nek.CASENAME}.re2",
                            f"{nek.CASENAME}.map",
                            f"{nek.CASENAME}.wall",
                            f"{nek.CASENAME}.restart",
                            
                            # Time Series Probs
                            f"stat_pts.in",
                            # Tripping 
                            f"forparam.i",
                            ],
                    'option':[
                            'SIZE',
                            f"mask_{nek.CASENAME}0.f00002",
                            ]
                    }
        for fname in checklist["must"]:
            from_file=os.path.join(self.nek.compile_path,fname)    
            to_file=os.path.join(self.rank_folder,fname)    
            if not os.path.exists(from_file): 
                raise FileNotFoundError(f"[IO] {from_file} not EXIST!")
            else: 
                
                # IF the file exist, we clean it to ensure everything works fine.
                if os.path.exists(to_file):
                    os.remove(to_file)
                    print(f'[IO] REMOVE EXIST: {to_file}')
                
                shutil.copy(from_file,to_file)
                print(f"[IO] {to_file} COPIED",flush=True)

        for fname in checklist["option"]:
            from_file=os.path.join(self.nek.compile_path,fname)    
            to_file=os.path.join(self.rank_folder,fname)    
            if not os.path.exists(to_file): 
                print(f"[IO] WARNING: {to_file} not EXIST",flush=True)
                shutil.copy(from_file,to_file)
            else: 
                print(f"[IO] {to_file} EXIST",flush=True)
                pass 
    
        return True 

    #-----------------------------------------
    def write_SESSION_NAME(self): 
        """Write the session name and where the code should be executed"""
        
        solver_root  =  Path(self.rank_folder)
        fileName  =  os.path.join(solver_root,'SESSION.NAME')
        is_exist  =  os.path.exists(fileName)
        if is_exist:
            os.remove(fileName)
        command  =  "cd %s && touch SESSION.NAME" % solver_root
        subprocess.call(command,shell = True)
        command  =  "cd %s && echo %s > SESSION.NAME" % (solver_root, self.nek.CASENAME) 
        subprocess.call(command,shell = True)
        command  =  "cd %s && echo $(pwd) >> SESSION.NAME" % solver_root 
        subprocess.call(command,shell = True)
        print('[IO] SESSION NAME WRITTEN',flush = True)
        return True

    #-----------------------------------------
    def rewrite_REA_v17(self):
        """
        Re-Write Parameter files for NEK verison < = 17
        For the controlable param, please see config.
        """
        # File Path 
        file_path = os.path.join(self.nek.compile_path,f"{self.nek.CASENAME}.rea")
        output_path = os.path.join(self.rank_folder,f'{self.nek.CASENAME}.rea')

        # Grab the original file and rewrite 
        with open(file_path, 'r') as f:
            lines = f.readlines()
            updated_lines = []
            for line in lines:
                if 'p' in line:
                    parts = line.split()
                    if len(parts) > 1 and parts[1] in param_mapping.values():
                        # Find the matching attribute
                        for attr, pkey in param_mapping.items():
                            if parts[1] == pkey:
                                # Update the value from the simulation object
                                parts[0] = f"{getattr(self.nek, attr):.6E}"
                                line = '\t'.join(parts) + '\n'
                                break
                updated_lines.append(line)
        f.close()

        # Update our re-written lines
        with open(output_path, 'w') as f:
            f.writelines(updated_lines)
        f.close()

        return True
    #-----------------------------------------
    def init_restart(self):
        """Copy the restart file to the target folder only if RSTART NOT EXIST"""

        restart_folder = os.path.join(self.nek.restart_folder,f"init_{self.drl.rank}")
        rs_list = os.listdir(restart_folder)
        rs_list = [f for f in rs_list if 'rs' in f ]
        
        rs_exist = os.listdir(self.rank_folder)
        rs_exist = [f for f in rs_exist if 'rs' in f ]
        
        if len(rs_exist) ==0:
            print(f"[INIT] IMPORTING RESTART FILES!",flush=True)
            for rsfile in rs_list: 
                rsfile = os.path.join(restart_folder,rsfile)
                shutil.copy(rsfile,dst=self.rank_folder)
                print(f"[STB3] RS6 file RESET: {rsfile}",flush=True)
                # print('[STB3] RS6 file RESET',flush=True)
        else:
            file_example = rs_exist[0]
            loc = file_example.find('rs')
            rsx = int(file_example[loc+2:loc+3])
            print(f"[INIT] {rsx} FILE!",flush=True)
            if len(rs_exist) < rsx//2:
                print(f"[INIT] {rsx} > {len(rs_list)}: IMPORTING RESTART FILES!",flush=True)
                for rsfile in rs_list: 
                    rsfile = os.path.join(restart_folder,rsfile)
                    shutil.copy(rsfile,dst=self.rank_folder)
                    print(f"[STB3] RS6 file RESET: {rsfile}",flush=True)
            else:
                print(f"[INIT] File Exists no need to copy!",flush=True)
                
                
        return True
    #-----------------------------------------
    def main(self):
        self.is_done.append(self.get_Case_Files())
        self.is_done.append(self.write_SESSION_NAME())
        self.is_done.append(self.rewrite_REA_v17())
        self.is_done.append(self.init_restart())
        
        if False not in self.is_done:
            return True
        else:
            return False


def remove_sch(current_path):
    file_list = os.listdir(current_path)
    file_list = [f for f in file_list if ".sch" in f]
    if len(file_list) > 0:
        for f in file_list:
            os.remove(os.path.join(current_path,f))
    return 


def oppo_control(observation,env):
    """Simple policy of applying the opposition CTRL"""
    actions = {}
    # Opposition control
    ## -1 ==> v-velocity 
    for agent in env.possible_agents:
        actions[agent]  =  -1.0*observation[agent][-1,0,0]
    return actions




def show_title():
    text_= """
--------------------------------------------------------
███╗   ██╗███████╗██╗  ██╗    ██████╗ ██████╗ ██╗     
████╗  ██║██╔════╝██║ ██╔╝    ██╔══██╗██╔══██╗██║     
██╔██╗ ██║█████╗  █████╔╝     ██║  ██║██████╔╝██║     
██║╚██╗██║██╔══╝  ██╔═██╗     ██║  ██║██╔══██╗██║     
██║ ╚████║███████╗██║  ██╗    ██████╔╝██║  ██║███████╗
            NEK5000 Reinforcement Learning
                Stable-Baselines3
                    Yuning Wang
--------------------------------------------------------

"""
    print(text_,flush=True)
    return

def show_end():
    text_= """
▗▖  ▗▖▗▄▄▄▖▗▖ ▗▖    ▗▄▄▄ ▗▄▄▖ ▗▖       ▗▄▄▄▖▗▖  ▗▖▗▄▄▄ 
▐▛▚▖▐▌▐▌   ▐▌▗▞▘    ▐▌  █▐▌ ▐▌▐▌       ▐▌   ▐▛▚▖▐▌▐▌  █
▐▌ ▝▜▌▐▛▀▀▘▐▛▚▖     ▐▌  █▐▛▀▚▖▐▌       ▐▛▀▀▘▐▌ ▝▜▌▐▌  █
▐▌  ▐▌▐▙▄▄▖▐▌ ▐▌    ▐▙▄▄▀▐▌ ▐▌▐▙▄▄▖    ▐▙▄▄▖▐▌  ▐▌▐▙▄▄▀
"""
    print(text_,flush=True)
    return