import subprocess
import sys
import os


SOURCE_FILE = "HIPProg.hip"

BASE_FLAGS = [
    "hipcc", "-x", "hip", SOURCE_FILE,
    "-Wno-c++11-narrowing", "-Wno-unused-value",
    "--offload-arch=native"
]


CMD_COMPILE_REF = BASE_FLAGS + ["-O0", "-o", "bin_ref"]


CMD_COMPILE_BUG = BASE_FLAGS + ["-O3", "-o", "bin_bug"]

# 60s timeout 
TIMEOUT = 60
"""
    Parses output to find CRC values.
    Returns sorted list to handle random thread printing order.
    """
def get_crc_values(output_str):
    
    values = []
    for line in output_str.split("\n"):
        line = line.strip()
        if "CRC:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                # stores the value part of the output
                values.append(parts[-1].strip())
    
    values.sort()
    return values

def run_variant(label, compile_cmd, binary_name):
    """Compiles and runs a variant. Returns stdout or None on failure."""
    
    # A. Compile
    res_comp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res_comp.returncode != 0:
        print(f"[{label}] Compile Failed")
        # print(res_comp.stderr) # Uncomment if you need to see compile errors
        return None

    # B. Run
    try:
        res_run = subprocess.run([f"./{binary_name}"], capture_output=True, text=True, timeout=TIMEOUT)
        if res_run.returncode != 0:
            print(f"[{label}] Runtime Crash (rc={res_run.returncode})")
            return None
        return res_run.stdout
    except subprocess.TimeoutError:
        print(f"[{label}] Runtime Timeout")
        return None

def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        sys.exit(1)

    print(f"Checking {SOURCE_FILE}...")

    # 1. Run Reference
    raw_ref = run_variant("REF (-O0)", CMD_COMPILE_REF, "bin_ref")
    if raw_ref is None:
        print(">> [FAIL] Reference failed to compile or run.")
        sys.exit(1)

    # 2. Run Bug
    raw_bug = run_variant("BUG (-O3)", CMD_COMPILE_BUG, "bin_bug")
    if raw_bug is None:
        print(">> [FAIL] Bug target crashed/failed.")
        sys.exit(1)

    # 3. Compare Logic
    vals_ref = get_crc_values(raw_ref)
    vals_bug = get_crc_values(raw_bug)

    if not vals_ref:
        print(">> [FAIL] No CRC data found in Reference output.")
        sys.exit(1)

    if vals_ref == vals_bug:
        # Code matches = Bug is lost
        print("\033[91m>> [FAIL] BUG LOST (Outputs Match)\033[0m")
        sys.exit(1)
    else:
        # Code differs = Bug is preserved
        print("-" * 40)
        print(f"Ref Sample: {vals_ref[0] if vals_ref else 'None'}")
        print(f"Bug Sample: {vals_bug[0] if vals_bug else 'None'}")
        print("-" * 40)
        print("\033[92m>> [SUCCESS] BUG PRESERVED (Outputs Differ)\033[0m")
        sys.exit(0)

if __name__ == "__main__":
    main()