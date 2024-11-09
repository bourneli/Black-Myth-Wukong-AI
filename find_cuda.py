# import torch

# def get_torch_cuda_info():
#     if torch.cuda.is_available():
#         cuda_version = torch.version.cuda
#         cudnn_version = torch.backends.cudnn.version()
#         cuda_available = True
#     else:
#         cuda_available = False
#         cuda_version = None
#         cudnn_version = None

#     return cuda_available, cuda_version, cudnn_version

# def get_cuda_toolkit_path():
#     try:
#         import ctypes
#         import re
#         cudart = ctypes.CDLL("cudart64_110.dll")
#         runtime_info = ctypes.create_string_buffer(4096)  # Adjust buffer size if necessary
#         cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_info))
#         runtime_info = runtime_info.value.decode('utf-8')
        
#         match = re.search(r'CUDNN\s+path\s+:\s+(.*)', runtime_info)
#         if match:
#             return match.group(1).strip()
#     except Exception as e:
#         print(f"Unable to retrieve CUDA toolkit path: {e}")
    
#     return None

# cuda_available, cuda_version, cudnn_version = get_torch_cuda_info()
# cuda_toolkit_path = get_cuda_toolkit_path()

# print(f"CUDA Available: {cuda_available}")
# print(f"CUDA Version: {cuda_version}")
# print(f"cuDNN Version: {cudnn_version}")
# print(f"CUDA Toolkit Path: {cuda_toolkit_path}")



# import os

# def find_cuda_via_env():
#     # Common environment variables where CUDA toolkit paths might be stored
#     env_vars = [
#         'CUDA_PATH',
#         'CUDA_PATH_V11_8',  # Adjust this according to your actual CUDA version
#         'CUDA_TOOLKIT_ROOT_DIR'
#     ]
    
#     for env_var in env_vars:
#         cuda_path = os.getenv(env_var)
#         if cuda_path and os.path.exists(cuda_path):
#             return cuda_path
            
#     return None

# def find_cudnn_via_cuda(cuda_path):
#     if not cuda_path:
#         return None

#     paths_to_check = [
#         os.path.join(cuda_path, 'include', 'cudnn.h'),
#         os.path.join(cuda_path, 'lib', 'x64', 'cudnn.lib')
#     ]

#     for path in paths_to_check:
#         if os.path.exists(path):
#             return os.path.dirname(path)

#     return None

# cuda_path = find_cuda_via_env()
# cudnn_path = find_cudnn_via_cuda(cuda_path)

# print(f"CUDA path: {cuda_path}")
# print(f"cuDNN path: {cudnn_path}")



import os
import winreg

def find_cuda_via_registry():
    try:
        # Open the registry key where CUDA might be registered
        reg_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA")
        num_versions = winreg.QueryInfoKey(reg_key)[0]

        versions = []
        for i in range(num_versions):
            try:
                version = winreg.EnumKey(reg_key, i)
                versions.append(version)
            except OSError:
                break

        # Assuming the latest version is to be used
        if versions:
            latest_version = sorted(versions, reverse=True)[0]
            cuda_path_key = winreg.OpenKey(reg_key, latest_version)
            cuda_path, _ = winreg.QueryValueEx(cuda_path_key, "InstallDir")
            return cuda_path
    except WindowsError as e:
        print(f"Windows registry query failed: {e}")
        return None

def find_cudnn_via_cuda(cuda_path):
    if not cuda_path:
        return None

    # Check common subdirectories for cuDNN files
    paths_to_check = [
        os.path.join(cuda_path, 'include', 'cudnn.h'),
        os.path.join(cuda_path, 'lib', 'x64', 'cudnn.lib')
    ]

    for path in paths_to_check:
        if os.path.exists(path):
            return os.path.dirname(path)

    return None

cuda_path = find_cuda_via_registry()
cudnn_path = find_cudnn_via_cuda(cuda_path)

print(f"CUDA path: {cuda_path}")
print(f"cuDNN path: {cudnn_path}")