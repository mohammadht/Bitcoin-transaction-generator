import tkinter as tk
from tkinter import ttk
import time
import threading
from tkinter import messagebox
import hashlib
import string
import os
import pandas as pd 
import numpy as np
import json
import random
from scipy.stats import zipf, yulesimon, planck, nbinom, poisson, geom, randint

def get_input_values():
    pattern = pattern_selection.get()  
    tx_count = tx_count_text_box.get()
    if pattern == "Real":
        # Get the real pattern as input and output
        input_dist = real_pattern_combo.get()  
        output_dist = real_pattern_combo.get()
        include_spikes = False # No need for fake spikes
    else: #if pattern is custom
        input_dist = input_count_combo.get()  # Get the selected value of the input count
        output_dist = output_count_combo.get()  # Get the selected value of the output count
        include_spikes = spikes_checkbox_var.get()  # Get the state of including spikes (True/False)
    include_utxo = utxo_checkbox_var.get() # Get the state of including UTXO generation
    print(f"Pattern: {pattern}, Transactions: {tx_count}, Input Dist: {input_dist}, Output Dist: {output_dist}, Include Spikes: {include_spikes}, Include UTXO: {include_utxo}")

    return pattern, tx_count, input_dist, output_dist, include_spikes, include_utxo

def run_task():
    # Get the input values before running the task
    pattern, tx_count, input_dist, output_dist, include_spikes, include_utxo = get_input_values()

    global running
    running = True
    progress_window = tk.Toplevel(root)
    progress_window.title("Processing")
    progress_window.geometry("400x150")

    progress_label = tk.Label(progress_window, text="Generating synthetic transactions, please wait...")
    progress_label.pack(pady=20)

    progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()

    # Add a Cancel button to stop the process
    cancel_button = tk.Button(progress_window, text="Cancel", command=lambda: cancel_task(progress_window))
    cancel_button.pack(pady=10)

    # Run the task in a separate thread
    threading.Thread(target=task, args=(progress_window, pattern, tx_count, input_dist, output_dist, include_spikes, include_utxo)).start()

def task(progress_window, pattern, tx_count, input_dist, output_dist, include_spikes, include_utxo):
    global running
    try:
        start_time = time.time()
        # Reading files based on input values
        if pattern == 'Real':
            io_file_path = f"DISTRIBUTIONS/REAL_IO_PROBABILITIES/io-{input_dist}.json"
            with open(io_file_path, 'r') as file_io:
                loaded_params_input_output = json.load(file_io)
        else: # pattern is custom
            input_file_path = f"DISTRIBUTIONS/input_count/input-{input_dist}.json"
            output_file_path = f"DISTRIBUTIONS/output_count/output-{output_dist}.json"
            loaded_params_input_output = []
            with open(input_file_path, 'r') as file_input:
                loaded_params_input_output.append(json.load(file_input))
            with open(output_file_path, 'r') as file_output:
                loaded_params_input_output.append(json.load(file_output))
        interval_file_path = f"DISTRIBUTIONS/bitcoin-interval.json"
        age_file_path = f"DISTRIBUTIONS/bitcoin-utxo-age.json"
        with open(interval_file_path, 'r') as file_interval:
            loaded_params_interval = json.load(file_interval)
        with open(age_file_path, 'r') as file_age:
            loaded_params_age = json.load(file_age)

        df_generated = generate_transactions(pattern, loaded_params_input_output, loaded_params_interval, int(tx_count), include_spikes)
        txs_file_path = f'results/bitcoin-generated-{tx_count}.tsv'
        df_generated.to_csv(txs_file_path, sep='\t', index=False)
        if include_utxo == True:
            df_utxo, df_spent_utxo = generate_utxos(txs_file_path, loaded_params_age)
            utxos_file_path = f'results/bitcoin-utxo-{tx_count}.tsv'
            df_utxo.to_csv(utxos_file_path, sep='\t', index=False)
            spent_utxo_file_path = f'results/bitcoin-inputs-{tx_count}.tsv'
            df_spent_utxo.to_csv(spent_utxo_file_path, sep='\t', index=False)
        end_time = time.time()
        duration = end_time - start_time
        print(f"All transactions generated and wrote to results/ directory in {duration:.2f} seconds.")

        if running:
            progress_window.destroy()
            messagebox.showinfo("Done!", f"All transactions generated and wrote to results/ directory in {duration:.2f} seconds.")
        else:
            progress_window.destroy()
    
    except Exception as e:
        progress_window.destroy()
        messagebox.showerror("Error", f"An error occurred: {e}")

def generate_transactions(pattern, loaded_params_input_output, loaded_params_interval, tx_count, include_spikes):
    print("Generating transactions...")
    start_txs = time.time()
    if pattern == "Real":
        glist_input, glist_output = generate_io(loaded_params_input_output, tx_count)
    else: #pattern is custom
        glist_input = generate_inputs(loaded_params_input_output[0], tx_count, include_spikes)
        glist_output = generate_outputs(loaded_params_input_output[1], tx_count, include_spikes)
    glist_size = generate_size(glist_input, glist_output)
    glist_interval = generate_intervals(loaded_params_interval, tx_count)
    glist_time = generate_time(glist_interval)
    glist_txhash = generate_txhash(tx_count)
    df_generated = pd.DataFrame({'tx_hash':glist_txhash, 'time_millisec':glist_time, 'interval':glist_interval, 'size':glist_size, 
                                 'input count': glist_input, 'output count':glist_output})
    df_generated['input/output ratio'] = df_generated['input count'] / df_generated['output count']
    end_txs = time.time()
    print(f"All txs generated in {end_txs - start_txs}.")
    return df_generated

def generate_io(loaded_params_io, tx_count):
    print("Generating input and output counts...")
    start_io = time.time()
    input_counts = np.array([item['input count'] for item in loaded_params_io])
    output_counts = np.array([item['output count'] for item in loaded_params_io])
    probs = np.array([item['probability'] for item in loaded_params_io])
    chosen_indices = np.random.choice(len(probs), size=tx_count, p=probs)
    glist_input = input_counts[chosen_indices]
    glist_output = output_counts[chosen_indices]
    end_io = time.time()
    print(f"{len(glist_input)} tx input and output counts generated in {end_io - start_io} seconds.")
    return glist_input, glist_output

def generate_inputs(loaded_params_input, tx_count, include_spikes):
    print("Generating input counts...")
    start_inputs = time.time()
    distribution_name = loaded_params_input['distribution']
    dist_input = eval(f'{distribution_name}')
    g_input = dist_input.rvs(loaded_params_input['params_0'], size=tx_count)
    glist_input = np.array(g_input)
    # Get the maximum allowed value from the loaded parameters
    max_value = loaded_params_input['max_value']
    # Identify indices where the generated values exceed the maximum allowed value
    exceeding_indices = np.where(glist_input > max_value)[0]
    # Replace exceeding values with new valid random values below the max_value
    if len(exceeding_indices) > 0:
        # Generate new random values that are below or equal to max_value
        replacement_values = dist_input.rvs(loaded_params_input['params_0'], size=len(exceeding_indices))
        glist_input[exceeding_indices] = replacement_values

    if include_spikes == True:
        # Define the specific values and their minimum occurrence probabilities
        specific_values_input = {
            10: 0.001,
            20: 0.0001,
            60: 0.0005,
            100: 0.0005,
            150: 0.0001,
            200: 0.0001,
            500: 0.0001,
            1000: 0.0001,
            50: 0.0001,
            70: 0.0001,
            80: 0.0001,
            250: 0.00001,
            300: 0.00001,
            400: 0.00001,
            600: 0.000001,
            800: 0.000001,
            1500: 0.000001
        }

        # Count occurrences of 9 and 19
        count_9_input = np.count_nonzero(glist_input == 9)
        count_19_input = np.count_nonzero(glist_input == 19)

        # Adjust the specific counts for 10 and 20 based on the desired ratio
        specific_values_input[10] = max(specific_values_input[10], 1.5 * count_9_input / tx_count)
        specific_values_input[20] = max(specific_values_input[20], 2 * count_19_input / tx_count)

        # Calculate the required number of occurrences for each value
        required_counts_input = {k: int(v * tx_count) for k, v in specific_values_input.items()}

        # Ensure each specific value occurs at least the required number of times
        for value, count in required_counts_input.items():
            current_count = np.count_nonzero(glist_input == value)
            if current_count < count:
                additional_count = count - current_count
                indices = np.random.choice(tx_count, additional_count, replace=False)
                for index in indices:
                    glist_input[index] = value
    end_inputs = time.time()
    print(f"{len(glist_input)} tx input counts generated in {end_inputs - start_inputs} seconds.")
    return glist_input

def generate_outputs(loaded_params_output, tx_count, include_spikes):
    print("Generating output counts...")
    start_outputs = time.time()
    distribution_name = loaded_params_output['distribution']
    dist_output = eval(f'{distribution_name}')
    g_output = dist_output.rvs(loaded_params_output['params_0'], size=tx_count)
    glist_output = np.array(g_output)
    # Get the maximum allowed value from the loaded parameters
    max_value = loaded_params_output['max_value']
    # Identify indices where the generated values exceed the maximum allowed value
    exceeding_indices = np.where(glist_output > max_value)[0]
    # Replace exceeding values with new valid random values below the max_value
    if len(exceeding_indices) > 0:
        # Generate new random values that are below or equal to max_value
        replacement_values = dist_output.rvs(loaded_params_output['params_0'], size=len(exceeding_indices))
        glist_output[exceeding_indices] = replacement_values
    
    if include_spikes == True:
        # Define the specific values and their minimum occurrence probabilities
        specific_values = {
            11: 0.001,
            21: 0.001,
            25: 0.001,
            31: 0.0005,
            51: 0.0005,
            101: 0.0005,
            201: 0.0001,
            251: 0.00005,
            301: 0.00005,
            401: 0.00001,
            601: 0.000001,
            701: 0.000001,
            801: 0.000001,
            501: 0.00005,
            1001: 0.00005,
            1501: 0.000001,
            2001: 0.000001,
        }

        # Count occurrences of 10, 20, 24, 30, 50
        count_10 = np.count_nonzero(glist_output == 10)
        count_20 = np.count_nonzero(glist_output == 20)
        count_24 = np.count_nonzero(glist_output == 24)
        count_30 = np.count_nonzero(glist_output == 30)
        count_50 = np.count_nonzero(glist_output == 50)

        # Adjust the specific counts for 11, 21, 25, 31, 51 based on the desired ratio
        specific_values[11] = max(specific_values[11], 1.5 * count_10 / tx_count)
        specific_values[21] = max(specific_values[21], 2 * count_20 / tx_count)
        specific_values[25] = max(specific_values[25], 2.5 * count_24 / tx_count)
        specific_values[31] = max(specific_values[31], 2 * count_30 / tx_count)
        specific_values[51] = max(specific_values[51], 2 * count_50 / tx_count)

        # Calculate the required number of occurrences for each value
        required_counts = {k: int(v * tx_count) for k, v in specific_values.items()}

        for value, count in required_counts.items():
            current_count = np.count_nonzero(glist_output == value)
            if current_count < count:
                additional_count = count - current_count
                indices = np.random.choice(tx_count, additional_count, replace=False)
                for index in indices:
                    glist_output[index] = value
    end_outputs = time.time()
    print(f"{len(glist_output)} tx output counts generated in {end_outputs - start_outputs} seconds.")
    return glist_output

# Function to load the preprocessed size data with cumulative probabilities
def load_cumulative_size_data(file_path):
    size_cumulative_prob_lookup = {}    
    # Load the data
    df_loaded = pd.read_csv(file_path, sep='\t', header=0)
    # Iterate through the rows and parse JSON data from the 'unique_sizes' and 'cumulative_probs' columns
    for _, row in df_loaded.iterrows():
        input_count = row['input count']
        output_count = row['output count']
        unique_sizes = json.loads(row['unique_sizes'])
        cumulative_probs = np.array(json.loads(row['cumulative_probs']))    
        # Store in the lookup dictionary
        size_cumulative_prob_lookup[(input_count, output_count)] = (unique_sizes, cumulative_probs)
    return size_cumulative_prob_lookup

# Function to find size
def find_size_with_cumulative_probabilities(generated_input, generated_output, size_cumulative_prob_lookup):
    key = (generated_input, generated_output)  
    if key in size_cumulative_prob_lookup:
        unique_sizes, cumulative_probs = size_cumulative_prob_lookup[key]    
        # Generate a random number and find the index in the cumulative probabilities
        rand_value = random.random()
        index = np.searchsorted(cumulative_probs, rand_value)   
        return unique_sizes[index]
    # Fallback in case no matching sizes are found
    return 306

def generate_size(glist_input, glist_output):
    print("Generating tx sizes...")
    start_size = time.time()
    size_cumulative_prob_lookup = load_cumulative_size_data('bitcoin-size-cumulative.tsv')
    glist_input_np = np.array(glist_input)
    glist_output_np = np.array(glist_output)
    glist_size = []
    for inp, out in zip(glist_input_np, glist_output_np):
        size = find_size_with_cumulative_probabilities(inp, out, size_cumulative_prob_lookup)
        glist_size.append(size)
    end_size = time.time()
    print(f"{len(glist_size)} tx sizes generated in {end_size - start_size} seconds.")
    return glist_size

def generate_intervals(loaded_params_interval, tx_count):
    print("Generating intervals...")
    start_intervals = time.time()
    glist_interval = []
    for i in range(4):
        distribution_name = loaded_params_interval[i]['distribution']
        param_0 = loaded_params_interval[i]['params_0']
        param_1 = loaded_params_interval[i].get('params_1')  # Use .get() to handle missing key
        size = int(tx_count * loaded_params_interval[i]['fraction'])
        min_value, max_value = loaded_params_interval[i]['min_value'], loaded_params_interval[i]['max_value']
        glist_interval.extend(generate_within_range(distribution_name, param_0, param_1, size, min_value, max_value))
    while(len(glist_interval) > tx_count):
        glist_interval.pop() 
    while(len(glist_interval) < tx_count):
        glist_interval.append(0)
    # Set the first tx inter-arrival time to 0
    first_element = 0
    # Shuffle the rest of the list
    remaining_elements = glist_interval[1:]
    random.shuffle(remaining_elements)
    # Recombine the first element with the shuffled rest
    glist_interval = [first_element] + remaining_elements
    end_intervals = time.time()
    print(f"{len(glist_interval)} tx inter-arrival time generated in {end_intervals - start_intervals} seconds.")
    return glist_interval

# Generate synthetic data based on known distributions within a defined range
def generate_within_range(distribution, param0, param1, size, min_val, max_val):
    generated = []
    while len(generated) < size:
        if distribution == 'nbinom':
            dist_time = eval(f'{distribution}')
            sample = dist_time.rvs(int(param0), param1, size=size)
        elif distribution == 'poisson' or 'geom' or 'randint' or 'planck':
            dist_interval = eval(f'{distribution}')
            sample = dist_interval.rvs(param0, param1, size=size)
        else:
            dist_time = eval(f'{distribution}')
            sample = dist_time.rvs(param0, size=size)
        sample = sample[(sample >= min_val) & (sample <= max_val)]
        generated.extend(sample)
    return generated[:size]

def generate_time(glist_interval):
    print("Generating tx times...")
    start_times = time.time()
    glist_time = []
    # Get the current time in milliseconds
    current_time_ms = int(time.time() * 1000)
    glist_time.append(current_time_ms)
    # Fill glist_time by adding the next transaction inter-arrival time to the previous timestamp
    for i in range(len(glist_interval) - 1):
        next_time = glist_time[-1] + glist_interval[i + 1]
        glist_time.append(next_time)
    end_times = time.time()
    print(f"{len(glist_time)} txs time generated in {end_times - start_times} seconds.")
    return glist_time

def generate_txhash(tx_count):
    print("Generating tx hashes...")
    start_hashes = time.time()
    # Use a set to ensure uniqueness
    glist_txhash = set()
    # Counter to ensure deterministic and unique inputs to the hash function
    counter = 0
    while len(glist_txhash) < tx_count:
        # Generate a unique hash using a counter value
        tx_hash = generate_deterministic_hash(counter)
        glist_txhash.add(tx_hash)  # Set ensures uniqueness
        counter += 1
    end_hashes = time.time()
    print(f"{len(glist_txhash)} tx hashes generated in {end_hashes - start_hashes} seconds.")        
    return list(glist_txhash)

def generate_deterministic_hash(counter):
    # Use the counter to create a unique string
    unique_string = f"transaction_{counter}"   
    # Create a SHA-256 hash of the unique string
    hash_object = hashlib.sha256(unique_string.encode())   
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()

def generate_random_hash():
    # Create a random string of characters
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=64))   
    # Create a SHA-256 hash of the random string
    hash_object = hashlib.sha256(random_string.encode())   
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()

def generate_random_utxo():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=34))

def generate_utxos(txs_file_path, loaded_params_age):
    print("Generating tx input and output files...")
    start_io = time.time()
    # Read the transactions from bitcoin-generated.tsv
    df = pd.read_csv(txs_file_path, sep='\t')
    # List to accumulate UTXO details
    utxo_list = []
    # input entries for bitcoin-inputs.tsv
    input_list = []
    # Dictionary for fast UTXO lookup by time (bucketed by time for faster lookups)
    utxo_by_time = {}
    # Store UTXO index in utxo_list for easy spent marking
    utxo_index_lookup = {}

    glist_age = generate_age(df['input count'].sum(),loaded_params_age)
    age_idx= 0

    # Process each row in the tx DataFrame
    for idx, row in df.iterrows():
        tx_hash = row['tx_hash']
        tx_time = row['time_millisec']
        time_index = row['time_millisec'] // 1000
        input_count = row['input count']
        output_count = row['output count']

        # Process outputs: generate utxo rows for bitcoin-utxo.tsv
        for _ in range(output_count):
            utxo = generate_random_utxo()
            utxo_list.append([utxo, tx_time, False])
            # Store UTXO by time for quick lookup
            if time_index not in utxo_by_time:
                utxo_by_time[time_index] = []
            
            # Track the index in utxo_list for spent status updating
            utxo_idx = len(utxo_list) - 1
            utxo_by_time[time_index].append((utxo, utxo_idx))
            utxo_index_lookup[(utxo, time_index)] = utxo_idx  # Track the index

        # Process inputs: generate input rows for bitcoin-inputs.tsv
        for _ in range(input_count):
            utxo_age = glist_age[age_idx]
            age_idx += 1
            
            # Calculate potential_utxo_time
            potential_utxo_time = tx_time // 1000 - utxo_age

            # Find a valid unspent UTXO
            found_utxo = None
            if potential_utxo_time in utxo_by_time:
                # Check for unspent UTXO in the bucket
                for i, (utxo, utxo_idx) in enumerate(utxo_by_time[potential_utxo_time]):
                    # Check if it's unspent by verifying against utxo_list
                    if not utxo_list[utxo_idx][2]:  # [2] is the 'spent' field in utxo_list
                        found_utxo = utxo
                        utxo_list[utxo_idx][2] = True  # Mark the corresponding UTXO as spent
                        break

            if found_utxo is None:
                found_utxo = generate_random_utxo()

            input_list.append([tx_hash, found_utxo, tx_time, utxo_age])

    utxo_df = pd.DataFrame(utxo_list, columns=['utxo', 'time_millisec', 'spent'])
    input_df = pd.DataFrame(input_list, columns=['tx_hash', 'spending_utxo', 'spending_time_millisec', 'utxo_age_sec'])
    end_io = time.time()
    print(f"Transaction input and output files generated in {end_io - start_io} seconds.")
    return utxo_df, input_df

def generate_age(txo_count, loaded_params_age):
    print("Generating utxo ages...")
    start_ages = time.time()
    glist_age = []
    keys_to_iterate = ["data_0_1","data_2_3","data_4_5","data_6_9", "data_10","data_100","data_1k",
    "data_10k","data_100k","data_1m","data_10m","data_100m"]
    for key in keys_to_iterate:
        distribution_name = loaded_params_age[key]['distribution']
        param_0 = loaded_params_age[key]['params_0']
        param_1 = loaded_params_age[key]['params_1']
        size = int(txo_count * loaded_params_age[key]['fraction'])
        min_value, max_value = loaded_params_age[key]['min_value'], loaded_params_age[key]['max_value']
        glist_age.extend(generate_within_range(distribution_name, param_0, param_1, size, min_value, max_value))
    while(len(glist_age) > txo_count):
        glist_age.pop()
    while(len(glist_age) < txo_count):
        glist_age.append(0)     
    random.shuffle(glist_age)
    end_ages = time.time()
    print(f"{len(glist_age)} UTXO ages generated in {end_ages - start_ages} seconds.")
    return glist_age

def cancel_task(progress_window):
    global running
    running = False
    progress_window.destroy()
    messagebox.showwarning("Cancelled", "Transaction generation was cancelled.")

def on_select_and_update_label(event, result_label, combo_box, file_type):
    selected_item = combo_box.get()    
    # Update the label with the content of the corresponding JSON file
    try:
        if file_type == 'input':
            file_path = f"DISTRIBUTIONS/input_count/{file_type}-{selected_item}.json"
        elif file_type == 'output': #file_type is output
            file_path = f"DISTRIBUTIONS/output_count/{file_type}-{selected_item}.json"
        else: #file_type is real
            file_path = f"DISTRIBUTIONS/REAL_IO_PROBABILITIES/{file_type}-{selected_item}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                loaded_params = json.load(file)
                # Update the label with the content
                if (file_type != "real"):
                    result_label.config(text=f'Distribution={loaded_params['distribution']} | Parameter={loaded_params['params_0']}')
        else:
            result_label.config(text="File not found.")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Function to handle the radio button selection
def update_pattern_selection():
    if pattern_selection.get() == "Real":
        root.geometry("600x250")
        # Enable real pattern combo box and disable custom pattern inputs
        real_pattern_combo.config(state="readonly")        
        # Show real pattern label and combo box
        real_pattern_label.grid()
        real_pattern_combo.grid()
        
        # Hide custom pattern inputs
        input_count_label.grid_remove()
        input_count_combo.grid_remove()
        result_input_count_label.grid_remove()
        output_count_label.grid_remove()
        output_count_combo.grid_remove()
        result_output_count_label.grid_remove()
        spikes_checkbox.grid_remove()
        spikes_checkbox_label.grid_remove()
    else:
        root.geometry("750x350")
        # Disable real pattern combo box and enable custom pattern inputs
        real_pattern_label.grid_remove()
        real_pattern_combo.grid_remove()
        
        # Show custom pattern inputs
        input_count_label.grid()
        input_count_combo.grid()
        result_input_count_label.grid()
        output_count_label.grid()
        output_count_combo.grid()
        result_output_count_label.grid()
        spikes_checkbox.grid()
        spikes_checkbox_label.grid()

# Validation function to allow only numeric input
def validate_numeric_input(P):
    return P.isdigit() or P == ""

# Main window setup
root = tk.Tk()
root.title("Bitcoin Transaction Generator")
root.geometry("800x350") #Window size
root.resizable(False, False)  # Prevent window resizing

# Register validation function
vcmd = (root.register(validate_numeric_input), '%P')

# Text box for number of transactions
tx_count_label = tk.Label(root, text="Number of Transactions:")
tx_count_label.grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)

tx_count_text_box = tk.Entry(root, validate='key', validatecommand=vcmd)
tx_count_text_box.grid(row=0, column=1, padx=5, pady=10, sticky=tk.W+tk.E)
# Set the default value of 1000
tx_count_text_box.insert(0, "1000")

# Radio Button for pattern selection
pattern_label = tk.Label(root, text="Transaction features based on...")
pattern_label.grid(row=1, column=0, padx=5, pady=10, sticky=tk.W)

# Radio button variables
pattern_selection = tk.StringVar(value="Real")  # Default to 'Real'

# Real patterns radio button
real_pattern_radio = tk.Radiobutton(root, text="Real patterns", variable=pattern_selection, value="Real", command=update_pattern_selection)
real_pattern_radio.grid(row=1, column=1, padx=5, pady=10, sticky=tk.W)

# Custom patterns radio button
custom_pattern_radio = tk.Radiobutton(root, text="Custom patterns", variable=pattern_selection, value="Custom", command=update_pattern_selection)
custom_pattern_radio.grid(row=1, column=2, padx=5, pady=10, sticky=tk.W)

# Real patterns combo box
real_pattern_label = tk.Label(root, text="Choose a real pattern:")
real_pattern_label.grid(row=2, column=0, padx=5, pady=10, sticky=tk.W)

real_pattern_combo = ttk.Combobox(root, values=["2023-Q4", "2023-Q3", "2023-Q2", "2023-Q1", "2022-Q4"], state="readonly")
real_pattern_combo.grid(row=2, column=1, padx=5, pady=10)
real_pattern_combo.set("2023-Q4")  # Set default selection

real_pattern_combo.bind("<<ComboboxSelected>>", lambda event: on_select_and_update_label(event, result_input_count_label, real_pattern_combo, "real"))

# Custom pattern options (input/output and spikes)
# Custom input
input_count_label = tk.Label(root, text="Input Count Distribution:")
input_count_label.grid(row=3, column=0, padx=5, pady=10, sticky=tk.W)

input_count_combo = ttk.Combobox(root, values=["custom-file", "2023-Q4", "2023-Q3", "2023-Q2", "2023-Q1", "2022-Q4"], state="readonly")
input_count_combo.grid(row=3, column=1, padx=5, pady=10)
input_count_combo.set("---select the pattern---")

result_input_count_label = tk.Label(root, text="(Adjust distribution parameters)")
result_input_count_label.grid(row=3, column=2, padx=5, pady=10, sticky=tk.W)

input_count_combo.bind("<<ComboboxSelected>>", lambda event: on_select_and_update_label(event, result_input_count_label, input_count_combo, "input"))

# Custom output
output_count_label = tk.Label(root, text="Output Count Distribution:")
output_count_label.grid(row=4, column=0, padx=5, pady=10, sticky=tk.W)

output_count_combo = ttk.Combobox(root, values=["custom-file", "2023-Q4", "2023-Q3"], state="readonly")
output_count_combo.grid(row=4, column=1, padx=5, pady=10)
output_count_combo.set("---select the pattern---")

result_output_count_label = tk.Label(root, text="(Adjust distribution parameters)")
result_output_count_label.grid(row=4, column=2, padx=5, pady=10, sticky=tk.W)

output_count_combo.bind("<<ComboboxSelected>>", lambda event: on_select_and_update_label(event, result_output_count_label, output_count_combo, "output"))

# Spikes Checkbox
spikes_checkbox_var = tk.BooleanVar(value=True)
spikes_checkbox = tk.Checkbutton(root, text="Include spikes in distributions", variable=spikes_checkbox_var)
spikes_checkbox.grid(row=5, column=0, padx=5, pady=10, sticky=tk.W)

spikes_checkbox_label = tk.Label(root, text="(Transaction consolidation and batching)")
spikes_checkbox_label.grid(row=5, column=1, padx=5, pady=10, sticky=tk.W)

# UTXO Checkbox
utxo_checkbox_var = tk.BooleanVar(value=True)
utxo_checkbox = tk.Checkbutton(root, text="Include UTXO generation", variable=utxo_checkbox_var)
utxo_checkbox.grid(row=6, column=0, padx=5, pady=10, sticky=tk.W)

utxo_checkbox_label = tk.Label(root, text="(Generate UTXO files too)")
utxo_checkbox_label.grid(row=6, column=1, padx=5, pady=10, sticky=tk.W)

# Generate Transactions Button
button = tk.Button(root, text="Generate Transactions", command=run_task)
button.grid(row=7, column=0, columnspan=3, pady=20)

# Initialize the state based on default radio button selection
update_pattern_selection()

root.mainloop()
