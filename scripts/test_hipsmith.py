import subprocess
import tempfile
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
ITERATIONS = 10000
CORES = 16
ERROR_LOG = "error_summary.log"

# Get the absolute path to HIPSmith so it can be called from temporary directories
HIPSMITH_PATH = os.path.abspath("./HIPSmith")
ROOT_DIR = os.path.abspath(".")

# The compilation command as a list of arguments
COMPILE_CMD = [
    "hipcc", "-x", "hip", "HIP-driver.cpp", "HIPProg.hip", "-I", ROOT_DIR,
    "-Werror=uninitialized", "-Werror=missing-field-initializers",
    "-Werror=array-bounds", "-Werror=zero-length-array",
    "-fno-strict-aliasing", "-Wno-c++11-narrowing", "-Wno-unused-value",
    "--offload-arch=native", "-o", "HIPProg"
]

def run_fuzz_iteration(i):
    """This function runs entirely inside an isolated temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate the HIP program
        try:
            gen_proc = subprocess.run([HIPSMITH_PATH, "--vectors", "--hip-consts"], 
                                      cwd=tmpdir, 
                                      stdout=subprocess.DEVNULL, 
                                      stderr=subprocess.DEVNULL)
            if gen_proc.returncode != 0:
                return (i, False, "HIPSmith generation failed", "")
        except Exception as e:
            return (i, False, f"HIPSmith execution error: {e}", "")

        # 2. Compile the generated program
        try:
            comp_proc = subprocess.run(COMPILE_CMD, 
                                       cwd=tmpdir, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.STDOUT, # Merge stderr into stdout
                                       text=True)
            
            compile_out = comp_proc.stdout
            exit_code = comp_proc.returncode
            
            # 3. Check for errors
            if exit_code != 0 or "error" in compile_out.lower():
                # We found an error! Read the failing code so we can pass it back.
                hip_file_path = os.path.join(tmpdir, "HIPProg.hip")
                hip_code = ""
                if os.path.exists(hip_file_path):
                    with open(hip_file_path, "r") as f:
                        hip_code = f.read()
                        
                return (i, True, compile_out, hip_code)
            
        except Exception as e:
            return (i, True, f"Compilation execution error: {e}", "")

    # Normal run, no errors
    return (i, False, "", "")

def draw_progress_bar(current, total):
    bar_length = 50
    percent = int(current * 100 / total)
    filled = int(percent * bar_length / 100)
    bar = '#' * filled + '-' * (bar_length - filled)
    # \r overwrites the current line
    sys.stdout.write(f"\rProgress: [{bar}] {percent}% ({current}/{total})")
    sys.stdout.flush()

def main():
    if not os.path.exists(HIPSMITH_PATH):
        print(f"[!] Error: Could not find HIPSmith at {HIPSMITH_PATH}")
        sys.exit(1)

    # Clear the previous error log
    with open(ERROR_LOG, "w") as f:
        f.write("")

    print(f"Starting parallel fuzzing loop on {CORES} cores for {ITERATIONS} iterations...")

    errors_found = 0

    # Start the thread pool
    with ProcessPoolExecutor(max_workers=CORES) as executor:
        # Submit all 10,000 tasks
        futures = {executor.submit(run_fuzz_iteration, i): i for i in range(1, ITERATIONS + 1)}
        
        completed = 0
        # as_completed yields tasks exactly as they finish
        for future in as_completed(futures):
            iteration, has_error, output, hip_code = future.result()
            
            if has_error:
                # Clear the progress bar line so text doesn't jumble
                sys.stdout.write("\r" + " " * 80 + "\r")
                print(f"[!] Error detected on iteration {iteration}")
                
                # Write to the error log
                with open(ERROR_LOG, "a") as f:
                    f.write(f"=== Error at Iteration {iteration} ===\n")
                    f.write(output + "\n\n")
                
                # Save the crashing HIP code into the main directory
                if hip_code:
                    crash_file = f"HIPProg_crash_{iteration}.hip"
                    with open(crash_file, "w") as f:
                        f.write(hip_code)
                    print(f"    -> Saved failing source to {crash_file}")
                errors_found += 1
                
            elif "HIPSmith generation failed" in output:
                sys.stdout.write("\r" + " " * 80 + "\r")
                print(f"[!] Iteration {iteration}: HIPSmith generation failed!")

            # Update Progress Bar
            completed += 1
            if completed % 10 == 0 or completed == ITERATIONS:
                draw_progress_bar(completed, ITERATIONS)

    print(f"\n\nDone! Found {errors_found} errors. Check {ERROR_LOG} for details.")

if __name__ == "__main__":
    main()