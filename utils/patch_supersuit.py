#!/usr/bin/env python3
"""
Automated script to patch Supersuit's vec_env_args function
This script modifies the Supersuit installation to fix the environment copying issue
"""

import os
import sys
import shutil
from pathlib import Path


def find_supersuit_path():
    """Find the Supersuit installation path"""
    try:
        import supersuit
        return os.path.dirname(supersuit.__file__)
    except ImportError:
        print("Error: Supersuit is not installed!")
        return None


def backup_original_file(file_path):
    """Create a backup of the original file"""
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")


def patch_vec_env_args(file_path):
    """Patch the vec_env_args function in vector_constructors.py"""
    
    # Read the current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define the original function (what we want to replace)
    original_function = """def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space"""
    
    # Define the patched function (what we want to replace it with)
    patched_function = """def vec_env_args(env, num_envs):
    if num_envs == 1:
        def env_fn():
            env_copy = env
            return env_copy
    else: 
        def env_fn():
            env_copy = cloudpickle.loads(cloudpickle.dumps(env))
            return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space"""
    
    # Check if the patch is already applied
    if patched_function in content:
        print("✅ Patch already applied!")
        return True
    
    # Check if the original function exists
    if original_function not in content:
        print("❌ Original function not found. File may have been modified.")
        return False
    
    # Apply the patch
    new_content = content.replace(original_function, patched_function)
    
    # Write the patched content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully applied Supersuit patch!")
    return True


def restore_original_file(file_path):
    """Restore the original file from backup"""
    backup_path = file_path + ".backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        print("✅ Restored original file from backup")
        return True
    else:
        print("❌ No backup file found")
        return False


def main():
    """Main function to handle patching operations"""
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        # Restore mode
        supersuit_path = find_supersuit_path()
        if supersuit_path:
            file_path = os.path.join(supersuit_path, "vector", "vector_constructors.py")
            restore_original_file(file_path)
        return
    
    # Patch mode (default)
    print("🔧 Patching Supersuit vec_env_args function...")
    
    supersuit_path = find_supersuit_path()
    if not supersuit_path:
        sys.exit(1)
    
    file_path = os.path.join(supersuit_path, "vector", "vector_constructors.py")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    print(f"📁 Supersuit path: {supersuit_path}")
    print(f"📄 Target file: {file_path}")
    
    # Create backup
    backup_original_file(file_path)
    
    # Apply patch
    if patch_vec_env_args(file_path):
        print("🎉 Supersuit patching completed successfully!")
        print("\nTo restore the original file, run:")
        print("python utils/patch_supersuit.py restore")
    else:
        print("❌ Patching failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
