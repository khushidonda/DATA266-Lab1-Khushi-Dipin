"""Task 1 (Dipin): device, hardware disclosure, peak-memory and raw-log helpers (Linux, Windows, macOS)."""

import json
import platform
import subprocess
from datetime import datetime

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def hardware_info(device):
    info = {"device": str(device), "cpu": cpu_name(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 2), cuda=torch.version.cuda)
    elif device.type == "mps":
        info["gpu"] = "Apple MPS (integrated GPU)"
    else:
        info["gpu"] = "none"
    return info


def peak_host_rss_bytes():
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(cb=ctypes.sizeof(PMC))
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if platform.system() == "Darwin" else rss * 1024


def peak_memory(device):
    """Peak accelerator memory (allocated / reserved) plus peak host RSS."""
    out = {"peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None,
           "peak_host_rss_bytes": peak_host_rss_bytes()}
    if device.type == "cuda":
        out["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        out["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    elif device.type == "mps":
        out["peak_gpu_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return out


def autocast(device, enabled):
    """bf16 autocast on CUDA when enabled; a no-op context elsewhere."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


class RawLogger:
    """Append-only JSONL log: one line per event, never rewritten."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "a", buffering=1, encoding="utf-8")

    def log(self, event, **kw):
        self.f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "event": event, **kw}) + "\n")

    def close(self):
        self.f.close()
