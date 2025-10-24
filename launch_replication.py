#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess
import time
import datetime

def setup_logging():
    """Set up automatic logging to results file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"replication_results_{timestamp}.txt"
    
    # Tee output to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Open log file and redirect stdout
    log_f = open(log_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(sys.stdout, log_f)
    
    print(f"Logging output to: {log_file}")
    print("=" * 60)
    print(f"EMOJI PAPER REPLICATION - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return log_f, original_stdout

def find_data_files():
    print("Looking for data files...")
    
    search_paths = [".", "data", "datasets", "..", "../..", "../../.."]
    parquet_patterns = ["output.parquet", "*.parquet"]
    csv_patterns = ["emojitweets.csv", "*emoji*.csv", "*tweet*.csv"]
    
    found_parquet = None
    found_csv = None
    
    for search_path in search_paths:
        for pattern in parquet_patterns:
            matches = list(Path(search_path).glob(pattern))
            if matches:
                found_parquet = matches[0]
                break
        if found_parquet:
            break
    
    for search_path in search_paths:
        for pattern in csv_patterns:
            matches = list(Path(search_path).glob(pattern))
            if matches:
                found_csv = matches[0]
                break
        if found_csv:
            break
    
    return found_parquet, found_csv

def interactive_setup():
    print("\nData File Setup")
    print("=" * 40)
    
    parquet_file, csv_file = find_data_files()
    
    if parquet_file:
        print(f"Found parquet file: {parquet_file}")
    else:
        print("No parquet file found")
        parquet_file = input("Enter path to output.parquet: ").strip()
        if not parquet_file:
            return None, None
    
    if csv_file:
        print(f"Found CSV file: {csv_file}")
    else:
        print("No CSV file found")
        csv_file = input("Enter path to emojitweets.csv: ").strip()
        if not csv_file:
            return None, None
    
    return parquet_file, csv_file

def choose_execution_mode():
    print("\nExecution Mode")
    print("=" * 40)
    print("Running Analysis")
    print("This will run analysis steps in sequence.")
    
    return "1"  # Always return full analysis mode

def run_script(script_name, args=None):
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    
    original_dir = Path.cwd()
    script_dir = Path(__file__).parent
    
    try:
        os.chdir(script_dir)
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        print(f"Running {script_name}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{script_name} completed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False
    finally:
        os.chdir(original_dir)

def run_script_simple(script_name, input_file=None, output_dir=None):
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    
    # Add arguments if provided
    if input_file:
        cmd.extend(["--input", str(input_file)])
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    
    print(f"Running {script_name}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def run_full_replication(parquet_file, csv_file):
    print("\nRunning Replication...")
    print("Estimated time: 30-60 minutes")
    
    output_dir = Path("replication_results")
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    failed_steps = []
    
    os.environ['INPUT_PARQUET'] = str(parquet_file)
    os.environ['INPUT_CSV'] = str(csv_file)
    os.environ['DATASET_DIR'] = str(output_dir / "dataset")
    os.environ['RESULTS_DIR'] = str(output_dir / "results")
    
    print("\nStep 1: Data Processing")
    print(f"Using parquet file: {parquet_file}")
    print(f"Using CSV file: {csv_file}")
    
    parquet_abs = Path(parquet_file).resolve()
    csv_abs = Path(csv_file).resolve()
    output_abs = Path(output_dir).resolve()
    
    success = run_script("build_dataset.py", [
        "--input", str(parquet_abs),
        "--outdir", str(output_abs / "dataset"),
        "--input-format", "parquet"
    ])
    if not success:
        failed_steps.append("data_processing")
        print("Data processing failed. Cannot continue with remaining steps.")
        return False
    
    print("\nStep 2: Baseline Models (Logistic Regression)")
    success = run_script("lr_comprehensive_evaluation.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("baseline_models")
    
    print("\nStep 3: Transformer Models")
    success = run_script("transformer_comprehensive_evaluation.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("transformer_models")
    
    print("\nStep 4: Learning Curves")
    success = run_script("compute_learning_curves.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("learning_curves")
    
    print("\nStep 5: Tokenizer Audit")
    success = run_script("tokenizer_audit.py", [
        "--outdir", str(output_abs / "results" / "tokenizer_audit")
    ])
    if not success:
        failed_steps.append("tokenizer_audit")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("REPLICATION COMPLETED!")
    print("=" * 50)
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Results saved to: {output_dir}")
    
    if failed_steps:
        print(f"Failed steps: {', '.join(failed_steps)}")
        print("Check error messages above for details")
        return False
    else:
        print("All steps completed successfully!")
        print("Check the results directory for outputs")
    return True



def main():
    # Set up automatic logging
    log_f, original_stdout = setup_logging()
    
    try:
        print("EMOJI PAPER REPLICATION LAUNCHER")
        print("=" * 50)
        print("This launcher will help you run the emoji paper replication")
        print("starting from your raw data files (output.parquet and emojitweets.csv)")
        print("=" * 50)
        
        # Setup data files
        parquet_file, csv_file = interactive_setup()
        if not parquet_file or not csv_file:
            print("Cannot proceed without data files")
            sys.exit(1)
        
        if not Path(parquet_file).exists():
            print(f"Parquet file not found: {parquet_file}")
            sys.exit(1)
        
        if not Path(csv_file).exists():
            print(f"CSV file not found: {csv_file}")
            sys.exit(1)
        
        choose_execution_mode()  # Just show the message
        
        success = run_full_replication(parquet_file, csv_file)
        
        if success:
            print("\n" + "=" * 50)
            print("REPLICATION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("Check the output directory for results")
            print("Review the results directory for detailed outputs")
        else:
            print("\nReplication failed. Check error messages above.")
            sys.exit(1)
    
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        log_f.close()
        print(f"\nLog file saved as: {log_f.name}")

if __name__ == "__main__":
    main()