#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess
import time
import datetime

def install_emoji_font():
    """Install emoji font for wordcloud generation on Linux systems"""
    print("Checking emoji font availability...")
    
    try:
        # Check if emoji fonts are already available
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
            '/System/Library/Fonts/Apple Color Emoji.ttc',
            'NotoColorEmoji.ttf',
            'NotoEmoji-VariableFont_wght.ttf'
        ]
        
        available_fonts = []
        for font_path in font_paths:
            if Path(font_path).exists():
                available_fonts.append(font_path)
                print(f"Found: {font_path}")
        
        if available_fonts:
            print(f"{len(available_fonts)} emoji font(s) available")
            return True
        
        # Try to install Noto Color Emoji font
        if sys.platform.startswith('linux'):
            print("No emoji fonts found. Installing Noto Color Emoji font...")
            
            # Try different package managers
            package_managers = [
                ['sudo', 'apt-get', 'update', '&&', 'sudo', 'apt-get', 'install', '-y', 'fonts-noto-color-emoji'],
                ['sudo', 'yum', 'install', '-y', 'google-noto-emoji-fonts'],
                ['sudo', 'dnf', 'install', '-y', 'google-noto-emoji-fonts'],
                ['sudo', 'pacman', '-S', 'noto-fonts-emoji']
            ]
            
            for cmd in package_managers:
                try:
                    print(f"Trying: {' '.join(cmd)}")
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        print("Emoji font installed successfully!")
                        return True
                    else:
                        print(f"Failed: {result.stderr}")
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            print("Could not install emoji font automatically.")
            print("Please install manually:")
            print("  Ubuntu/Debian: sudo apt-get install fonts-noto-color-emoji")
            print("  CentOS/RHEL: sudo yum install google-noto-emoji-fonts")
            print("  Fedora: sudo dnf install google-noto-emoji-fonts")
            print("  Arch: sudo pacman -S noto-fonts-emoji")
            return False
            
        elif sys.platform == 'darwin':
            print("Detected macOS - emoji fonts should be available by default")
            return True
        elif sys.platform.startswith('win'):
            print("Detected Windows - emoji fonts should be available by default")
            return True
        else:
            print(f"Unknown platform: {sys.platform}")
            return False
            
    except Exception as e:
        print(f"Error checking emoji fonts: {e}")
        return False

def setup_logging():
    """Set up automatic logging to results file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"replication_results_{timestamp}.txt"
    
    # Create a custom print function that writes to both console and file
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
    print("Running Full Analysis (1-4 hours, all analyses)")
    print("This will run all analysis steps in sequence.")
    
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
    """Run a script with simplified arguments"""
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    
    # Create a simple command that works for most scripts
    cmd = [sys.executable, str(script_path)]
    
    # Add common arguments if provided
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
    print("\nRunning Full Replication...")
    print("Estimated time: 1-4 hours")
    
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
    
    print("\nStep 2: Percentile Analysis")
    success = run_script("text_emoji_length_percentiles.py", [
        "--input", str(output_abs / "dataset" / "text_emoji.csv"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("percentiles")
    
    print("\nStep 3: Emoji Sentiment Analysis")
    success = run_script("emoji_count_sentiment_distribution.py", [
        "--dataset", str(output_abs / "dataset" / "text_emoji.csv"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("emoji_sentiment")
    
    print("\nStep 4: Baseline Models (Logistic Regression)")
    success = run_script("lr_comprehensive_evaluation.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("baseline_models")
    
    print("\nStep 5: Transformer Models")
    success = run_script("transformer_comprehensive_evaluation.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("transformer_models")
    
    print("\nStep 6: Sequence Length Analysis")
    success = run_script("sequence_length_sensitivity.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("sequence_length")
    
    print("\nStep 7: Emoji Wordclouds")
    print("Checking emoji font availability for wordcloud generation...")
    install_emoji_font()
    success = run_script("emoji_wordcloud.py", [
        "--input", str(output_abs / "dataset" / "text_emoji.csv"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("wordclouds")
    
    print("\nStep 8: Learning Curves")
    success = run_script("compute_learning_curves.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("learning_curves")
    
    print("\nStep 9: Computational Analysis")
    success = run_script("computational_efficiency_analysis.py", [
        "--dataset-dir", str(output_abs / "dataset"),
        "--output-dir", str(output_abs / "results")
    ])
    if not success:
        failed_steps.append("computational")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("FULL REPLICATION COMPLETED!")
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
        print("This launcher will help you run the complete emoji paper replication")
        print("starting from your raw data files (output.parquet and emojitweets.csv)")
        print("=" * 50)
        
        # Get data files
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
            print("Review replication_report.md for details")
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