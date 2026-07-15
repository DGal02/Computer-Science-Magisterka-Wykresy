import os
import csv
import json
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic plotting (LaTeX style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})
sns.set_palette("muted")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def parse_mem(mem_str):
    """Parse memory string (e.g., '41.16MiB / 7.754GiB') and return value in MiB."""
    try:
        part = mem_str.split('/')[0].strip()
        if not part:
            return 0.0
        
        # Check suffix
        if part.endswith('GiB') or part.endswith('GB'):
            val = float(part[:-3].strip())
            return val * 1024.0
        elif part.endswith('MiB') or part.endswith('MB'):
            val = float(part[:-3].strip())
            return val
        elif part.endswith('KiB') or part.endswith('KB'):
            val = float(part[:-3].strip())
            return val / 1024.0
        elif part.endswith('B'):
            val = float(part[:-1].strip())
            return val / (1024.0 * 1024.0)
        else:
            return float(part)
    except Exception:
        return 0.0

def parse_cpu(cpu_str):
    """Parse CPU string (e.g., '303.11%') and return float."""
    try:
        return float(cpu_str.strip().replace('%', ''))
    except Exception:
        return 0.0

def parse_io_value(val_str):
    """Parse disk IO string value (e.g. '15.3MB') and return value in MiB."""
    try:
        val_str = val_str.strip().lower()
        if not val_str or val_str == "0b" or val_str == "0 b":
            return 0.0
        # check suffix
        if val_str.endswith('gib') or val_str.endswith('gb'):
            val = float(val_str[:-3].strip() if val_str.endswith('gib') else val_str[:-2].strip())
            return val * 1024.0
        elif val_str.endswith('mib') or val_str.endswith('mb'):
            val = float(val_str[:-3].strip() if val_str.endswith('mib') else val_str[:-2].strip())
            return val
        elif val_str.endswith('kib') or val_str.endswith('kb'):
            val = float(val_str[:-3].strip() if val_str.endswith('kib') else val_str[:-2].strip())
            return val / 1024.0
        elif val_str.endswith('b'):
            val = float(val_str[:-1].strip())
            return val / (1024.0 * 1024.0)
        else:
            return float(val_str)
    except Exception:
        return 0.0

def parse_block_io(io_str):
    """Parse Block IO string (e.g. '15.3MB / 0B') and return (read_mib, write_mib) tuple."""
    try:
        parts = io_str.split('/')
        if len(parts) == 2:
            read_val = parse_io_value(parts[0])
            write_val = parse_io_value(parts[1])
            return read_val, write_val
    except Exception:
        pass
    return 0.0, 0.0

def get_stats_file_name(category):
    if category.startswith("mix_"):
        return "stats_mix.csv"
    return f"stats_{category}.csv"

def aggregate_results():
    print(f"Starting data aggregation from: {DATA_DIR}")
    
    # Delete old aggregated CSVs
    for old_csv in glob.glob(os.path.join(os.path.dirname(__file__), "aggregated_results_*.csv")):
        try:
            os.remove(old_csv)
        except Exception:
            pass

    records = []

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return

    # Traverse plots/data/{system}/{host_system}/{run_id}
    for system_name in os.listdir(DATA_DIR):
        system_path = os.path.join(DATA_DIR, system_name)
        if not os.path.isdir(system_path):
            continue

        for host_sys in os.listdir(system_path):
            host_path = os.path.join(system_path, host_sys)
            if not os.path.isdir(host_path):
                continue

            for run_id in os.listdir(host_path):
                run_path = os.path.join(host_path, run_id)
                if not os.path.isdir(run_path):
                    continue

                # Read metadata.json
                meta_file = os.path.join(run_path, "metadata.json")
                max_cpus = "unknown"
                host_system = host_sys  # Default to directory name, update if metadata has it
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                            max_cpus = meta.get("max_cpus", max_cpus)
                            host_system = meta.get("host_system", host_system)
                    except Exception as e:
                        print(f"Warning: Failed to read metadata at {meta_file}: {e}")

                # Traverse payload sizes (small, medium)
                for payload_size in os.listdir(run_path):
                    payload_path = os.path.join(run_path, payload_size)
                    if not os.path.isdir(payload_path) or payload_size not in ["small", "medium"]:
                        continue

                    # Traverse iteration directories (0, 10000, 100000, 1000000)
                    for it_name in os.listdir(payload_path):
                        it_path = os.path.join(payload_path, it_name)
                        if not os.path.isdir(it_path):
                            continue
                        try:
                            iterations = int(it_name)
                        except ValueError:
                            continue

                        # Traverse workers-X directories
                        for workers_name in os.listdir(it_path):
                            workers_path = os.path.join(it_path, workers_name)
                            if not os.path.isdir(workers_path) or not workers_name.startswith("workers-"):
                                continue
                            try:
                                workers_count = int(workers_name.split("-")[1])
                            except ValueError:
                                continue

                            # Read operations in workers-X directory
                            files = os.listdir(workers_path)
                            csv_files = [f for f in files if f.endswith(".csv") and not f.startswith("stats_")]
                            
                            categories = set()
                            for f in csv_files:
                                parts = f.rsplit("-", 1)
                                if len(parts) == 2:
                                    categories.add(parts[0])

                            for cat in sorted(categories):
                                cat_files = [f for f in csv_files if f.startswith(f"{cat}-")]
                                 
                                all_latencies_arrays = []
                                sum_elapsed_by_worker = []
                                total_ops = 0

                                for cf in cat_files:
                                    cf_path = os.path.join(workers_path, cf)
                                    try:
                                        df_csv = pd.read_csv(cf_path, usecols=['elapsed_ms'], dtype={'elapsed_ms': np.float32})
                                        latencies = df_csv['elapsed_ms'].values
                                        if len(latencies) > 0:
                                            all_latencies_arrays.append(latencies)
                                            sum_elapsed_by_worker.append(np.sum(latencies))
                                            total_ops += len(latencies)
                                    except Exception as e:
                                        print(f"Warning: Failed to parse {cf_path}: {e}")

                                if not all_latencies_arrays:
                                    continue

                                max_worker_time = max(sum_elapsed_by_worker) if sum_elapsed_by_worker else 0
                                throughput = (total_ops / max_worker_time * 1000.0) if max_worker_time > 0 else 0.0

                                lat_arr = np.concatenate(all_latencies_arrays)
                                p50 = np.percentile(lat_arr, 50)
                                p95 = np.percentile(lat_arr, 95)
                                p99 = np.percentile(lat_arr, 99)
                                mean_lat = np.mean(lat_arr)
                                min_lat = np.min(lat_arr)
                                max_lat = np.max(lat_arr)

                                cpu_db_mean = 0.0
                                cpu_client_mean = 0.0
                                mem_db_mean = 0.0
                                mem_client_mean = 0.0
                                io_db_read_max = 0.0
                                io_db_write_max = 0.0
                                io_client_read_max = 0.0
                                io_client_write_max = 0.0

                                stats_file = os.path.join(workers_path, get_stats_file_name(cat))
                                if os.path.exists(stats_file):
                                    db_cpus = []
                                    client_cpus = []
                                    db_mems = []
                                    client_mems = []
                                    db_io_read = []
                                    db_io_write = []
                                    client_io_read = []
                                    client_io_write = []
                                    
                                    try:
                                        with open(stats_file, "r") as sf:
                                            reader = csv.DictReader(sf)
                                            for row in reader:
                                                container = row.get("container", "")
                                                cpu_val = parse_cpu(row.get("cpu_perc", "0.0%"))
                                                mem_val = parse_mem(row.get("mem_usage", "0B"))
                                                read_io, write_io = parse_block_io(row.get("block_io", "0B / 0B"))
                                                
                                                if container == "python-app":
                                                    client_cpus.append(cpu_val)
                                                    client_mems.append(mem_val)
                                                    client_io_read.append(read_io)
                                                    client_io_write.append(write_io)
                                                elif container.startswith("system-"):
                                                    db_cpus.append(cpu_val)
                                                    db_mems.append(mem_val)
                                                    db_io_read.append(read_io)
                                                    db_io_write.append(write_io)
                                        
                                        num_samples = len(db_cpus) if db_cpus else len(client_cpus)
                                        if num_samples > 0:
                                            if db_cpus:
                                                cpu_db_mean = sum(db_cpus) / len(db_cpus)
                                            if client_cpus:
                                                cpu_client_mean = sum(client_cpus) / len(client_cpus)
                                            if db_mems:
                                                mem_db_mean = sum(db_mems) / len(db_mems)
                                            if client_mems:
                                                mem_client_mean = sum(client_mems) / len(client_mems)
                                            if db_io_read:
                                                io_db_read_max = max(db_io_read) - min(db_io_read)
                                            if db_io_write:
                                                io_db_write_max = max(db_io_write) - min(db_io_write)
                                            if client_io_read:
                                                io_client_read_max = max(client_io_read) - min(client_io_read)
                                            if client_io_write:
                                                io_client_write_max = max(client_io_write) - min(client_io_write)
                                    except Exception as e:
                                        print(f"Warning: Failed to parse stats file {stats_file}: {e}")

                                records.append({
                                    "system": system_name,
                                    "run_id": run_id,
                                    "max_cpus": max_cpus,
                                    "host_system": host_system,
                                    "payload_size": payload_size,
                                    "iterations": iterations,
                                    "workers": workers_count,
                                    "category": cat,
                                    "total_ops": total_ops,
                                    "throughput_ops_sec": throughput,
                                    "latency_mean_ms": mean_lat,
                                    "latency_p50_ms": p50,
                                    "latency_p95_ms": p95,
                                    "latency_p99_ms": p99,
                                    "latency_min_ms": min_lat,
                                    "latency_max_ms": max_lat,
                                    "cpu_db_mean_perc": cpu_db_mean,
                                    "cpu_client_mean_perc": cpu_client_mean,
                                    "mem_db_mean_mib": mem_db_mean,
                                    "mem_client_mean_mib": mem_client_mean,
                                    "io_db_read_mib": io_db_read_max,
                                    "io_db_write_mib": io_db_write_max,
                                    "io_client_read_mib": io_client_read_max,
                                    "io_client_write_mib": io_client_write_max,
                                    "resource_samples": num_samples
                                })

    # Write aggregated CSVs per host_system
    df = pd.DataFrame(records)
    if not df.empty:
        for host_val, group in df.groupby("host_system"):
            host_clean = str(host_val).strip().lower().replace("/", "_").replace("\\", "_")
            csv_path = os.path.join(os.path.dirname(__file__), f"aggregated_results_{host_clean}.csv")
            group.to_csv(csv_path, index=False)
            print(f"Aggregation complete. Saved {len(group)} configurations to: {csv_path}")
    else:
        print("No configurations found to aggregate.")

def generate_plots():
    # Find all aggregated CSV files matching aggregated_results_*.csv
    summary_files = glob.glob(os.path.join(os.path.dirname(__file__), "aggregated_results_*.csv"))
    
    if not summary_files:
        print("Error: No aggregated results files matching aggregated_results_*.csv found. Please run 'aggregate' first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    skipped_resource_plots = []

    for summary_file in summary_files:
        basename = os.path.basename(summary_file)
        # Extract host system name from aggregated_results_{host_system}.csv
        host_sys = basename[len("aggregated_results_"):-4]
        
        df = pd.read_csv(summary_file)
        print(f"\nProcessing aggregated file: {basename} ({len(df)} configurations)")

        # Set styling
        sns.set_theme(style="whitegrid")
        crud_ops = ["insert", "read", "update", "delete"]
        
        op_translation = {
            "insert": "Zapis (Insert)",
            "read": "Odczyt (Read)",
            "update": "Aktualizacja (Update)",
            "delete": "Usuwanie (Delete)"
        }
        
        systems = df["system"].unique()

        for sys in systems:
            sys_df = df[df["system"] == sys]
            print(f"Processing system: {sys} on host: {host_sys}")

            # Create system and host-specific output directory
            sys_output_dir = os.path.join(OUTPUT_DIR, sys, host_sys)
            os.makedirs(sys_output_dir, exist_ok=True)

            # Get unique payload sizes and iterations
            payloads = sys_df["payload_size"].unique()

            for payload in payloads:
                p_df = sys_df[sys_df["payload_size"] == payload]
                its = p_df["iterations"].unique()

                for it in its:
                    if it == 0:
                        continue  # handled separately for advanced scenarios

                    # Only plot resource utilization for the largest operations:
                    # - 1M for small payload
                    # - 100k for medium payload
                    is_largest_small = (payload == "small" and it == 1000000)
                    is_largest_medium = (payload == "medium" and it == 100000)
                    if not (is_largest_small or is_largest_medium):
                        continue

                    workload_df = p_df[p_df["iterations"] == it]
                    payload_label = "128 B" if payload == "small" else "4 KB"
                    it_label = f"{it//1000}k" if it < 1000000 else "1M"

                    # Get host system for labels
                    host_sys_val = workload_df["host_system"].dropna().iloc[0] if "host_system" in workload_df.columns and not workload_df["host_system"].dropna().empty else host_sys
                    host_label = f"Środowisko: {str(host_sys_val).upper()}"

                    # Plot 1: Throughput scaling plots disabled (data consolidated in LaTeX performance tables)

                    # -------------------------------------------------------------
                    # Plot 3: CPU, RAM & Disk Utilization in time (insert operation - 2x2 Grid timeline)
                    # -------------------------------------------------------------
                    resource_data = workload_df[workload_df["category"] == "insert"]
                    if not resource_data.empty:
                        max_cpu_val = resource_data["max_cpus"].max()
                        max_workers = resource_data[resource_data["max_cpus"] == max_cpu_val]["workers"].max()
                        best_run = resource_data[(resource_data["max_cpus"] == max_cpu_val) & (resource_data["workers"] == max_workers)]
                        
                        if not best_run.empty:
                            samples_count = best_run["resource_samples"].iloc[0]
                            if samples_count < 30:
                                skipped_resource_plots.append({
                                    "system": sys.upper(),
                                    "host": host_sys.upper(),
                                    "payload": "128 B" if payload == "small" else "4 KB",
                                    "iterations": f"{it//1000}k" if it < 1000000 else "1M",
                                    "cpus": int(max_cpu_val),
                                    "workers": int(max_workers),
                                    "operation": "insert",
                                    "samples": int(samples_count)
                                })
                            
                            if samples_count >= 30:
                                run_id = best_run["run_id"].iloc[0]
                                stats_file = os.path.join(DATA_DIR, sys, host_sys, run_id, payload, str(it), f"workers-{max_workers}", "stats_insert.csv")
                                
                                if os.path.exists(stats_file):
                                    try:
                                        df_stats = pd.read_csv(stats_file)
                                        df_db = df_stats[df_stats["container"].str.startswith("system-")].copy()
                                        df_client = df_stats[df_stats["container"] == "python-app"].copy()
                                        
                                        if not df_db.empty or not df_client.empty:
                                            start_time = None
                                            
                                            if not df_db.empty:
                                                df_db["time_dt"] = pd.to_datetime(df_db["timestamp"])
                                                df_db = df_db.sort_values("time_dt")
                                                start_time = df_db["time_dt"].min()
                                                df_db["elapsed_sec"] = (df_db["time_dt"] - start_time).dt.total_seconds()
                                                df_db["cpu"] = df_db["cpu_perc"].apply(parse_cpu)
                                                df_db["ram"] = df_db["mem_usage"].apply(parse_mem)
                                                db_io = df_db["block_io"].apply(parse_block_io)
                                                df_db["io_read"] = [r for r, w in db_io]
                                                df_db["io_write"] = [w for r, w in db_io]
                                                df_db["io_read_delta"] = df_db["io_read"] - df_db["io_read"].iloc[0]
                                                df_db["io_write_delta"] = df_db["io_write"] - df_db["io_write"].iloc[0]
                                                
                                            if not df_client.empty:
                                                df_client["time_dt"] = pd.to_datetime(df_client["timestamp"])
                                                df_client = df_client.sort_values("time_dt")
                                                if start_time is None:
                                                    start_time = df_client["time_dt"].min()
                                                df_client["elapsed_sec"] = (df_client["time_dt"] - start_time).dt.total_seconds()
                                                df_client["cpu"] = df_client["cpu_perc"].apply(parse_cpu)
                                                df_client["ram"] = df_client["mem_usage"].apply(parse_mem)
                                                client_io = df_client["block_io"].apply(parse_block_io)
                                                df_client["io_read"] = [r for r, w in client_io]
                                                df_client["io_write"] = [w for r, w in client_io]
                                                df_client["io_read_delta"] = df_client["io_read"] - df_client["io_read"].iloc[0]
                                                df_client["io_write_delta"] = df_client["io_write"] - df_client["io_write"].iloc[0]
                                                
                                            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                                            ax_cpu, ax_ram = axes[0, 0], axes[0, 1]
                                            ax_read, ax_write = axes[1, 0], axes[1, 1]
                                            
                                            # 1. CPU timeline
                                            if not df_db.empty:
                                                ax_cpu.plot(df_db["elapsed_sec"], df_db["cpu"], 'o', label="Baza danych (serwer)", color="firebrick", markersize=3)
                                            if not df_client.empty:
                                                ax_cpu.plot(df_client["elapsed_sec"], df_client["cpu"], 'o', label="Aplikacja Python (klient)", color="royalblue", markersize=3)
                                            ax_cpu.set_title("Zużycie procesora (CPU) w czasie", fontsize=12, fontweight='bold')
                                            ax_cpu.set_ylabel("Zużycie CPU [%]")
                                            ax_cpu.set_xlabel("Czas trwania testu [s]")
                                            ax_cpu.grid(True, linestyle="--", alpha=0.5)
                                            ax_cpu.legend()
                                            
                                            # 2. RAM timeline
                                            if not df_db.empty:
                                                ax_ram.plot(df_db["elapsed_sec"], df_db["ram"], 'o', label="Baza danych (serwer)", color="firebrick", markersize=3)
                                            if not df_client.empty:
                                                ax_ram.plot(df_client["elapsed_sec"], df_client["ram"], 'o', label="Aplikacja Python (klient)", color="royalblue", markersize=3)
                                            ax_ram.set_title("Zużycie pamięci RAM w czasie", fontsize=12, fontweight='bold')
                                            ax_ram.set_ylabel("Zużycie pamięci RAM [MiB]")
                                            ax_ram.set_xlabel("Czas trwania testu [s]")
                                            ax_ram.grid(True, linestyle="--", alpha=0.5)
                                            ax_ram.legend()
                                            
                                            # 3. Disk Read timeline (cumulative delta)
                                            if not df_db.empty:
                                                ax_read.plot(df_db["elapsed_sec"], df_db["io_read_delta"], 'o', label="Baza danych (serwer)", color="firebrick", markersize=3)
                                            if not df_client.empty:
                                                ax_read.plot(df_client["elapsed_sec"], df_client["io_read_delta"], 'o', label="Aplikacja Python (klient)", color="royalblue", markersize=3)
                                            ax_read.set_title("Odczyt danych z dysku w czasie (Block I/O)", fontsize=12, fontweight='bold')
                                            ax_read.set_ylabel("Skumulowany odczyt z dysku [MiB]")
                                            ax_read.set_xlabel("Czas trwania testu [s]")
                                            ax_read.grid(True, linestyle="--", alpha=0.5)
                                            ax_read.legend()
                                            
                                            # 4. Disk Write timeline (cumulative delta)
                                            if not df_db.empty:
                                                ax_write.plot(df_db["elapsed_sec"], df_db["io_write_delta"], 'o', label="Baza danych (serwer)", color="firebrick", markersize=3)
                                            if not df_client.empty:
                                                ax_write.plot(df_client["elapsed_sec"], df_client["io_write_delta"], 'o', label="Aplikacja Python (klient)", color="royalblue", markersize=3)
                                            ax_write.set_title("Zapis danych na dysk w czasie (Block I/O)", fontsize=12, fontweight='bold')
                                            ax_write.set_ylabel("Skumulowany zapis na dysk [MiB]")
                                            ax_write.set_xlabel("Czas trwania testu [s]")
                                            ax_write.grid(True, linestyle="--", alpha=0.5)
                                            ax_write.legend()
                                            
                                            plt.suptitle(f"{sys.upper()}: Przebieg użycia zasobów w czasie (Zapis, {host_label})\n(Konfiguracja: {max_cpu_val} CPU, {max_workers} procesów klienckich, rozmiar rekordu: {payload_label}, {it_label} operacji/klient)", fontsize=14, fontweight='bold')
                                            plt.tight_layout()
                                            
                                            fig.savefig(os.path.join(sys_output_dir, f"resource_utilization_{payload}_{it}.pdf"), format="pdf", bbox_inches="tight")
                                            plt.close(fig)
                                            print(f"Generated: {sys}/{host_sys}/resource_utilization_{payload}_{it} (PDF)")
                                    except Exception as e:
                                        print(f"Warning: Failed to generate resource utilization timeline for {stats_file}: {e}")

            # -------------------------------------------------------------
            # Plot 4: Payload comparison (Small vs Medium) combined on subplots
            # Plotted for 1M small, 100k medium, max CPU (4) and max workers (8 or 1)
            # -------------------------------------------------------------
            df_crud_all = sys_df[sys_df["category"].isin(crud_ops)]
            
            df_small_all = df_crud_all[df_crud_all["payload_size"] == "small"]
            df_medium_all = df_crud_all[df_crud_all["payload_size"] == "medium"]
            
            df_small_plot = pd.DataFrame()
            df_medium_plot = pd.DataFrame()
            
            if not df_small_all.empty:
                # Find maximum iterations (preferably 1M)
                max_it_small = df_small_all["iterations"].max()
                df_small_sub = df_small_all[df_small_all["iterations"] == max_it_small]
                # Find maximum CPUs (preferably 4)
                max_cpu_small = df_small_sub["max_cpus"].max()
                df_small_sub = df_small_sub[df_small_sub["max_cpus"] == max_cpu_small]
                # Find maximum workers (preferably 8)
                max_workers_small = df_small_sub["workers"].max()
                df_small_plot = df_small_sub[df_small_sub["workers"] == max_workers_small]
                
            if not df_medium_all.empty:
                # Find maximum iterations (preferably 100k)
                max_it_medium = df_medium_all["iterations"].max()
                df_medium_sub = df_medium_all[df_medium_all["iterations"] == max_it_medium]
                # Find maximum CPUs (preferably 4)
                max_cpu_medium = df_medium_sub["max_cpus"].max()
                df_medium_sub = df_medium_sub[df_medium_sub["max_cpus"] == max_cpu_medium]
                # Find maximum workers (preferably 8)
                max_workers_medium = df_medium_sub["workers"].max()
                df_medium_plot = df_medium_sub[df_medium_sub["workers"] == max_workers_medium]
                
            df_payload_comp = pd.concat([df_small_plot, df_medium_plot])
            
            if not df_payload_comp.empty and len(df_payload_comp["payload_size"].unique()) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                df_plot = df_payload_comp.copy()
                df_plot["Rozmiar ładunku (Payload)"] = df_plot["payload_size"].map({
                    "small": "Mały (128 B)",
                    "medium": "Średni (4 KB)"
                })
                df_plot["Operacja"] = df_plot["category"].map(op_translation)
                
                sns.barplot(
                    data=df_plot,
                    x="Operacja",
                    y="throughput_ops_sec",
                    hue="Rozmiar ładunku (Payload)",
                    ax=ax,
                    edgecolor="0.2",
                    errorbar=None
                )
                
                it_s_label = f"{int(max_it_small)//1000}k" if max_it_small < 1000000 else "1M"
                it_m_label = f"{int(max_it_medium)//1000}k" if max_it_medium < 1000000 else "1M"
                
                # Check CPU and worker configurations to display in title
                cpu_label = f"{int(max_cpu_small)} CPU"
                workers_label = f"{int(max_workers_small)} kl." if max_workers_small == max_workers_medium else f"kl.: S:{int(max_workers_small)}/M:{int(max_workers_medium)}"
                
                ax.set_title(f"Porównanie przepustowości: {it_s_label} (Mały) vs {it_m_label} (Średni)\n(Konfiguracja: {cpu_label}, {workers_label})", fontsize=11, fontweight='bold')
                ax.set_ylabel("Przepustowość [operacji/s]")
                ax.set_xlabel("Operacja bazodanowa")
                ax.grid(True, linestyle="--", alpha=0.5, axis='y')
                
                for container in ax.containers:
                    labels = [f'{int(v):,}' for v in container.datavalues]
                    ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=9)
                
                host_sys_val = sys_df["host_system"].dropna().iloc[0] if "host_system" in sys_df.columns and not sys_df["host_system"].dropna().empty else host_sys
                host_label = f"Środowisko: {str(host_sys_val).upper()}"
                plt.suptitle(f"Wpływ rozmiaru rekordu na przepustowość {sys.upper()} ({host_label})", fontsize=13, fontweight='bold', y=0.98)
                plt.tight_layout()
                fig.savefig(os.path.join(sys_output_dir, "payload_comparison.pdf"), format="pdf", bbox_inches="tight")
                plt.close(fig)
                print(f"Generated: {sys}/{host_sys}/payload_comparison (PDF)")

            # -------------------------------------------------------------
            # Plot 5: Advanced/Fixed categories comparison (mix, queue, json_doc) combined on subplots
            # -------------------------------------------------------------
            fixed_ops_map = {
                "mix_50W_50R": "Mieszany (50/50)",
                "mix_90W_10R": "Mieszany (90W/10R)",
                "mix_10W_90R": "Mieszany (10W/90R)",
                "queue": "Kolejka (push/pop)",
                "doc_insert": "Dokument JSON: Zapis",
                "doc_read": "Dokument JSON: Odczyt",
                "doc_read_partial": "Dokument JSON: Odczyt częściowy",
                "doc_update_partial": "Dokument JSON: Aktualizacja",
                "doc_increment": "Dokument JSON: Inkrementacja",
                "doc_delete": "Dokument JSON: Usuwanie"
            }

            valid_payloads = []
            for payload in payloads:
                max_cpu_val = sys_df[sys_df["iterations"] == 0]["max_cpus"].max()
                max_workers = sys_df[sys_df["iterations"] == 0]["workers"].max()
                df_advanced = sys_df[
                    (sys_df["payload_size"] == payload) &
                    (sys_df["max_cpus"] == max_cpu_val) &
                    (sys_df["workers"] == max_workers) &
                    (sys_df["category"].isin(fixed_ops_map.keys()))
                ]
                if not df_advanced.empty:
                    valid_payloads.append((payload, max_cpu_val, max_workers))
            valid_payloads = sorted(valid_payloads, key=lambda x: x[0])
            
            if valid_payloads:
                fig, axes = plt.subplots(1, len(valid_payloads), figsize=(8 * len(valid_payloads), 6), squeeze=False)
                axes = axes.flatten()
                
                for idx, (payload, max_cpu_val, max_workers) in enumerate(valid_payloads):
                    ax = axes[idx]
                    df_advanced = sys_df[
                        (sys_df["payload_size"] == payload) &
                        (sys_df["max_cpus"] == max_cpu_val) &
                        (sys_df["workers"] == max_workers) &
                        (sys_df["category"].isin(fixed_ops_map.keys()))
                    ]
                    df_plot = df_advanced.copy()
                    df_plot["Typ testu"] = df_plot["category"].map(fixed_ops_map)
                    df_plot = df_plot.dropna(subset=["Typ testu"])
                    df_plot = df_plot.sort_values(by="throughput_ops_sec", ascending=False)
                    
                    if not df_plot.empty:
                        sns.barplot(
                            data=df_plot,
                            x="throughput_ops_sec",
                            y="Typ testu",
                            ax=ax,
                            hue="Typ testu",
                            palette="viridis",
                            legend=False,
                            edgecolor="0.2",
                            errorbar=None
                        )
                        payload_label = "128 B" if payload == "small" else "4 KB"
                        ax.set_title(f"Rozmiar ładunku: {payload_label}\n(Konfiguracja: {max_cpu_val} CPU, {max_workers} kl.)", fontsize=11, fontweight='bold')
                        ax.set_xlabel("Przepustowość [operacji/s]")
                        ax.set_ylabel("Scenariusz testowy")
                        ax.grid(True, linestyle="--", alpha=0.5, axis='x')
                        
                        for container in ax.containers:
                            labels = [f' {int(v):,}' for v in container.datavalues]
                            ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=9)
                            
                host_sys_val = sys_df["host_system"].dropna().iloc[0] if "host_system" in sys_df.columns and not sys_df["host_system"].dropna().empty else host_sys
                host_label = f"Środowisko: {str(host_sys_val).upper()}"
                plt.suptitle(f"Przepustowość w zaawansowanych scenariuszach {sys.upper()} ({host_label})", fontsize=13, fontweight='bold', y=0.98)
                plt.tight_layout()
                fig.savefig(os.path.join(sys_output_dir, "advanced_scenarios.pdf"), format="pdf", bbox_inches="tight")
                plt.close(fig)
                print(f"Generated: {sys}/{host_sys}/advanced_scenarios (PDF)")

            # -------------------------------------------------------------
            # Generate Performance Tables (Throughput and Latency; CSV and LaTeX format)
            # -------------------------------------------------------------
            for payload in payloads:
                p_df = sys_df[sys_df["payload_size"] == payload]
                its = p_df["iterations"].unique()
                
                for it in its:
                    if it == 0:
                        continue
                    
                    workload_df = p_df[p_df["iterations"] == it]
                    lat_data = workload_df[workload_df["category"].isin(crud_ops)].copy()
                    
                    if not lat_data.empty:
                        lat_data["category_pl"] = lat_data["category"].map(op_translation)
                        
                        table_df = lat_data[[
                            "category_pl", "max_cpus", "workers", 
                            "throughput_ops_sec",
                            "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"
                        ]].copy()
                        table_df = table_df.sort_values(by=["category_pl", "max_cpus", "workers"])
                        
                        csv_df = table_df.rename(columns={
                            "category_pl": "Operacja",
                            "max_cpus": "CPU Limit",
                            "workers": "Procesy Klienckie",
                            "throughput_ops_sec": "Przepustowosc [ops/s]",
                            "latency_mean_ms": "Srednia [ms]",
                            "latency_p50_ms": "Mediana (p50) [ms]",
                            "latency_p95_ms": "Percentyl 95 (p95) [ms]",
                            "latency_p99_ms": "Percentyl 99 (p99) [ms]"
                        })
                        csv_path = os.path.join(sys_output_dir, f"latency_table_{payload}_{it}.csv")
                        csv_df.to_csv(csv_path, index=False)
                        print(f"Generated table: {sys}/{host_sys}/latency_table_{payload}_{it}.csv")
                        
                        tex_path = os.path.join(sys_output_dir, f"latency_table_{payload}_{it}.tex")
                        payload_label = "128 B" if payload == "small" else "4 KB"
                        it_label = f"{it//1000}k" if it < 1000000 else "1M"
                        
                        with open(tex_path, "w", encoding="utf-8") as f:
                            f.write("% Tabela wygenerowana automatycznie\n")
                            f.write("\\begin{table}[ht]\n")
                            f.write("\\centering\n")
                            f.write("\\resizebox{\\textwidth}{!}{%\n")
                            f.write("\\begin{tabular}{llcrrrrr}\n")
                            f.write("\\toprule\n")
                            f.write("Operacja & CPU & Klienci & Przepustowość [op./s] & Średnia [ms] & Mediana (p50) [ms] & p95 [ms] & p99 [ms] \\\\\n")
                            f.write("\\midrule\n")
                            
                            for _, row in table_df.iterrows():
                                f.write(f"{row['category_pl']} & {row['max_cpus']} CPU & {row['workers']} & "
                                        f"{row['throughput_ops_sec']:.1f} & {row['latency_mean_ms']:.4f} & {row['latency_p50_ms']:.4f} & "
                                        f"{row['latency_p95_ms']:.4f} & {row['latency_p99_ms']:.4f} \\\\\n")
                            
                            f.write("\\bottomrule\n")
                            f.write("\\end{tabular}%\n")
                            f.write("}\n")
                            f.write(f"\\caption{{Wyniki wydajnościowe i opóźnienia operacji bazy danych {sys.upper()} (Rozmiar rekordu: {payload_label}, {it_label} operacji/klient)}}\n")
                            f.write(f"\\label{{tab:latency_{sys}_{payload}_{it}}}\n")
                            f.write("\\end{table}\n")
                        print(f"Generated LaTeX code: {sys}/{host_sys}/latency_table_{payload}_{it}.tex")

            # -------------------------------------------------------------
            # Generate Resource Utilization Tables (CSV and LaTeX format)
            # -------------------------------------------------------------
            for payload in payloads:
                p_df = sys_df[sys_df["payload_size"] == payload]
                its = p_df["iterations"].unique()
                
                for it in its:
                    workload_df = p_df[p_df["iterations"] == it]
                    res_data = workload_df[workload_df["resource_samples"] > 0].copy()
                    
                    if not res_data.empty:
                        all_ops_translation = {**op_translation, **fixed_ops_map}
                        res_data["category_pl"] = res_data["category"].map(all_ops_translation).fillna(res_data["category"])
                        
                        is_embedded = sys.lower() in ["berkeleydb", "leveldb", "rocksdb"]
                        
                        if is_embedded:
                            table_df = res_data[[
                                "category_pl", "max_cpus", "workers", 
                                "cpu_client_mean_perc", "mem_client_mean_mib",
                                "io_client_read_mib", "io_client_write_mib",
                                "resource_samples"
                            ]].copy()
                            table_df = table_df.sort_values(by=["category_pl", "max_cpus", "workers"])
                            
                            csv_df = table_df.rename(columns={
                                "category_pl": "Operacja",
                                "max_cpus": "CPU Limit",
                                "workers": "Procesy Klienckie",
                                "cpu_client_mean_perc": "CPU Klient [%]",
                                "mem_client_mean_mib": "RAM Klient [MiB]",
                                "io_client_read_mib": "Odczyt Klient [MiB]",
                                "io_client_write_mib": "Zapis Klient [MiB]",
                                "resource_samples": "Liczba probek"
                            })
                            csv_path = os.path.join(sys_output_dir, f"resource_table_{payload}_{it}.csv")
                            csv_df.to_csv(csv_path, index=False)
                            print(f"Generated table: {sys}/{host_sys}/resource_table_{payload}_{it}.csv")
                            
                            tex_path = os.path.join(sys_output_dir, f"resource_table_{payload}_{it}.tex")
                            payload_label = "128 B" if payload == "small" else "4 KB"
                            it_label = f"{it//1000}k" if it > 0 else "zaawansowane"
                            
                            with open(tex_path, "w", encoding="utf-8") as f:
                                f.write("% Tabela wygenerowana automatycznie\n")
                                f.write("\\begin{table}[ht]\n")
                                f.write("\\centering\n")
                                f.write("\\resizebox{\\textwidth}{!}{%\n")
                                f.write("\\begin{tabular}{llcrrcc}\n")
                                f.write("\\toprule\n")
                                f.write("Operacja & CPU & Klienci & CPU Kl. [\\%] & RAM Kl. [MiB] & We/Wy Kl. [MiB] & Próbki \\\\\n")
                                f.write("\\midrule\n")
                                
                                for _, row in table_df.iterrows():
                                    client_io = f"{row['io_client_read_mib']:.1f}/{row['io_client_write_mib']:.1f}"
                                    f.write(f"{row['category_pl']} & {row['max_cpus']} CPU & {row['workers']} & "
                                            f"{row['cpu_client_mean_perc']:.1f}\\% & {row['mem_client_mean_mib']:.1f} & "
                                            f"{client_io} & {int(row['resource_samples'])} \\\\\n")
                                            
                                f.write("\\bottomrule\n")
                                f.write("\\end{tabular}%\n")
                                f.write("}\n")
                                f.write(f"\\caption{{Zużycie zasobów kontenera aplikacji klienckiej (wbudowany silnik {sys.upper()}, Rozmiar rekordu: {payload_label}, {it_label})}}\n")
                                f.write(f"\\label{{tab:resources_{sys}_{payload}_{it}}}\n")
                                f.write("\\end{table}\n")
                            print(f"Generated LaTeX code: {sys}/{host_sys}/resource_table_{payload}_{it}.tex")
                        else:
                            table_df = res_data[[
                                "category_pl", "max_cpus", "workers", 
                                "cpu_db_mean_perc", "mem_db_mean_mib", 
                                "cpu_client_mean_perc", "mem_client_mean_mib",
                                "io_db_read_mib", "io_db_write_mib",
                                "resource_samples"
                            ]].copy()
                            table_df = table_df.sort_values(by=["category_pl", "max_cpus", "workers"])
                            
                            csv_df = table_df.rename(columns={
                                "category_pl": "Operacja",
                                "max_cpus": "CPU Limit",
                                "workers": "Procesy Klienckie",
                                "cpu_db_mean_perc": "CPU Baza [%]",
                                "mem_db_mean_mib": "RAM Baza [MiB]",
                                "cpu_client_mean_perc": "CPU Klient [%]",
                                "mem_client_mean_mib": "RAM Klient [MiB]",
                                "io_db_read_mib": "Odczyt Baza [MiB]",
                                "io_db_write_mib": "Zapis Baza [MiB]",
                                "resource_samples": "Liczba probek"
                            })
                            csv_path = os.path.join(sys_output_dir, f"resource_table_{payload}_{it}.csv")
                            csv_df.to_csv(csv_path, index=False)
                            print(f"Generated table: {sys}/{host_sys}/resource_table_{payload}_{it}.csv")
                            
                            tex_path = os.path.join(sys_output_dir, f"resource_table_{payload}_{it}.tex")
                            payload_label = "128 B" if payload == "small" else "4 KB"
                            it_label = f"{it//1000}k" if it > 0 else "zaawansowane"
                            
                            with open(tex_path, "w", encoding="utf-8") as f:
                                f.write("% Tabela wygenerowana automatycznie\n")
                                f.write("\\begin{table}[ht]\n")
                                f.write("\\centering\n")
                                f.write("\\resizebox{\\textwidth}{!}{%\n")
                                f.write("\\begin{tabular}{llcrrrrcc}\n")
                                f.write("\\toprule\n")
                                f.write("Operacja & CPU & Klienci & CPU DB [\\%] & RAM DB [MiB] & CPU Kl. [\\%] & RAM Kl. [MiB] & We/Wy DB [MiB] & Próbki \\\\\n")
                                f.write("\\midrule\n")
                                
                                for _, row in table_df.iterrows():
                                    db_io = f"{row['io_db_read_mib']:.1f}/{row['io_db_write_mib']:.1f}"
                                    f.write(f"{row['category_pl']} & {row['max_cpus']} CPU & {row['workers']} & "
                                            f"{row['cpu_db_mean_perc']:.1f}\\% & {row['mem_db_mean_mib']:.1f} & "
                                            f"{row['cpu_client_mean_perc']:.1f}\\% & {row['mem_client_mean_mib']:.1f} & "
                                            f"{db_io} & {int(row['resource_samples'])} \\\\\n")
                                            
                                f.write("\\bottomrule\n")
                                f.write("\\end{tabular}%\n")
                                f.write("}\n")
                                f.write(f"\\caption{{Zużycie zasobów kontenerów {sys.upper()} (Rozmiar rekordu: {payload_label}, {it_label})}}\n")
                                f.write(f"\\label{{tab:resources_{sys}_{payload}_{it}}}\n")
                                f.write("\\end{table}\n")
                            print(f"Generated LaTeX code: {sys}/{host_sys}/resource_table_{payload}_{it}.tex")

            # -------------------------------------------------------------
            # Generate Execution Summary (CSV and LaTeX format)
            # -------------------------------------------------------------
            all_ops_translation = {**op_translation, **fixed_ops_map}
            summary_records = []
            grouped = sys_df.groupby(["category", "payload_size", "iterations"])
            
            for (cat, payload, it), group in grouped:
                executed_count = len(group[["max_cpus", "workers"]].drop_duplicates())
                expected_count = 12
                status = "Kompletny" if executed_count == expected_count else f"{executed_count}/{expected_count}"
                
                cat_pl = all_ops_translation.get(cat, cat)
                payload_pl = "128 B" if payload == "small" else "4 KB"
                it_label = f"{it//1000}k" if it > 0 else "zaawansowane"
                
                summary_records.append({
                    "Operacja": cat_pl,
                    "Rozmiar rekordu": payload_pl,
                    "Liczba operacji": it_label,
                    "Wykonane konfiguracje": status,
                    "Procent ukończenia": f"{(executed_count / expected_count) * 100:.1f}%"
                })
                
            summary_df = pd.DataFrame(summary_records)
            summary_df = summary_df.sort_values(by=["Liczba operacji", "Rozmiar rekordu", "Operacja"])
            
            # Save CSV
            csv_path = os.path.join(sys_output_dir, "executed_tests_summary.csv")
            summary_df.to_csv(csv_path, index=False)
            print(f"Generated summary table: {sys}/{host_sys}/executed_tests_summary.csv")
            
            # Save LaTeX
            tex_path = os.path.join(sys_output_dir, "executed_tests_summary.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write("% Tabela wygenerowana automatycznie\n")
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\resizebox{\\textwidth}{!}{%\n")
                f.write("\\begin{tabular}{lllcc}\n")
                f.write("\\toprule\n")
                f.write("Scenariusz testowy & Rozmiar rekordu & Typ obciążenia & Wykonano & Ukończono [\\%] \\\\\n")
                f.write("\\midrule\n")
                
                for _, row in summary_df.iterrows():
                    pct_esc = str(row['Procent ukończenia']).replace("%", "\\%")
                    f.write(f"{row['Operacja']} & {row['Rozmiar rekordu']} & {row['Liczba operacji']} & "
                            f"{row['Wykonane konfiguracje']} & {pct_esc} \\\\\n")
                            
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}%\n")
                f.write("}\n")
                f.write(f"\\caption{{Podsumowanie wykonanych testów wydajnościowych dla systemu {sys.upper()} (Suma konfiguracji CPU i procesów klienckich)}}\n")
                f.write(f"\\label{{tab:executed_tests_{sys}}}\n")
                f.write("\\end{table}\n")
            print(f"Generated LaTeX code: {sys}/{host_sys}/executed_tests_summary.tex")

        # Generate comparative tables across all databases for the current host system
        print(f"\nGenerating comparative tables for host system: {host_sys}...")
        systems_list = ['redis', 'memcached', 'leveldb', 'rocksdb', 'couchbase', 'tarantool', 'foundationdb', 'berkeleydb']
        sys_display = {
            'redis': 'Redis',
            'memcached': 'Memcached',
            'leveldb': 'LevelDB*',
            'rocksdb': 'RocksDB*',
            'couchbase': 'Couchbase',
            'tarantool': 'Tarantool',
            'foundationdb': 'FoundationDB',
            'berkeleydb': 'BerkeleyDB'
        }
        
        # 1. Comparative Performance Table
        rows_perf = []
        for sys_name in systems_list:
            sys_df = df[(df['system'] == sys_name) & (df['payload_size'] == 'small') & (df['max_cpus'] == 4)]
            if sys_df.empty:
                continue
                
            workers_val = 1 if sys_name in ['leveldb', 'rocksdb'] else 8
            sys_df = sys_df[sys_df['workers'] == workers_val]
            
            sys_row = {'system': sys_display[sys_name]}
            for cat in ['insert', 'read', 'update', 'delete']:
                cat_df = sys_df[sys_df['category'] == cat]
                if not cat_df.empty:
                    thr = cat_df['throughput_ops_sec'].iloc[0]
                    p99 = cat_df['latency_p99_ms'].iloc[0]
                    sys_row[f'{cat}_thr'] = f"{thr:.1f}"
                    sys_row[f'{cat}_p99'] = f"{p99:.3f}"
                else:
                    sys_row[f'{cat}_thr'] = "n/a"
                    sys_row[f'{cat}_p99'] = "n/a"
            rows_perf.append(sys_row)
            
        if rows_perf:
            perf_df = pd.DataFrame(rows_perf)
            perf_tex_path = os.path.join(os.path.dirname(__file__), "..", "magisterka", f"tabelka_porownawcza_wydajnosc_{host_sys}.tex")
            with open(perf_tex_path, "w", encoding="utf-8") as f:
                f.write("% Tabela wygenerowana automatycznie\n")
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\resizebox{\\textwidth}{!}{%\n")
                f.write("\\begin{tabular}{lcrrccrrcc}\n")
                f.write("\\toprule\n")
                f.write(" & \\multicolumn{2}{c}{\\textbf{Zapis (Insert)}} & \\multicolumn{2}{c}{\\textbf{Odczyt (Read)}} & \\multicolumn{2}{c}{\\textbf{Aktualizacja (Update)}} & \\multicolumn{2}{c}{\\textbf{Usuwanie (Delete)}} \\\\\n")
                f.write("\\cmidrule(r){2-3} \\cmidrule(r){4-5} \\cmidrule(r){6-7} \\cmidrule(r){8-9}\n")
                f.write("Baza danych & Przep. [op./s] & p99 [ms] & Przep. [op./s] & p99 [ms] & Przep. [op./s] & p99 [ms] & Przep. [op./s] & p99 [ms] \\\\\n")
                f.write("\\midrule\n")
                for _, row in perf_df.iterrows():
                    f.write(f"{row['system']} & {row['insert_thr']} & {row['insert_p99']} & "
                            f"{row['read_thr']} & {row['read_p99']} & {row['update_thr']} & {row['update_p99']} & "
                            f"{row['delete_thr']} & {row['delete_p99']} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}%\n")
                f.write("}\n")
                f.write(f"\\caption{{Porównanie przepustowości i opóźnień p99 dla operacji CRUD (Rekord 128 B, środowisko {host_sys.upper()}, 4 CPU, 8 procesów klienckich, *dla LevelDB/RocksDB 1 proces)}}\n")
                f.write(f"\\label{{tab:porownanie_crud_{host_sys}}}\n")
                f.write("\\end{table}\n")
            print(f"Generated comparative performance table: {perf_tex_path}")
            
        # 2. Comparative Resource Table
        rows_res = []
        for sys_name in systems_list:
            sys_df = df[(df['system'] == sys_name) & (df['payload_size'] == 'small') & (df['max_cpus'] == 4)]
            if sys_df.empty:
                continue
                
            workers_val = 1 if sys_name in ['leveldb', 'rocksdb'] else 8
            sys_df = sys_df[sys_df['workers'] == workers_val]
            
            sys_row = {'system': sys_display[sys_name]}
            for cat in ['insert', 'read']:
                cat_df = sys_df[sys_df['category'] == cat]
                if not cat_df.empty:
                    cpu_db = cat_df['cpu_db_mean_perc'].iloc[0]
                    mem_db = cat_df['mem_db_mean_mib'].iloc[0]
                    io_read = cat_df['io_db_read_mib'].iloc[0]
                    io_write = cat_df['io_db_write_mib'].iloc[0]
                    sys_row[f'{cat}_cpu'] = f"{cpu_db:.1f}\\%"
                    sys_row[f'{cat}_mem'] = f"{mem_db:.1f}"
                    sys_row[f'{cat}_io'] = f"{io_read:.1f}/{io_write:.1f}"
                else:
                    sys_row[f'{cat}_cpu'] = "n/a"
                    sys_row[f'{cat}_mem'] = "n/a"
                    sys_row[f'{cat}_io'] = "n/a"
            rows_res.append(sys_row)
            
        if rows_res:
            res_df = pd.DataFrame(rows_res)
            res_tex_path = os.path.join(os.path.dirname(__file__), "..", "magisterka", f"tabelka_porownawcza_zasoby_{host_sys}.tex")
            with open(res_tex_path, "w", encoding="utf-8") as f:
                f.write("% Tabela wygenerowana automatycznie\n")
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\resizebox{\\textwidth}{!}{%\n")
                f.write("\\begin{tabular}{lcrccrcc}\n")
                f.write("\\toprule\n")
                f.write(" & \\multicolumn{3}{c}{\\textbf{Zapis (Insert)}} & \\multicolumn{3}{c}{\\textbf{Odczyt (Read)}} \\\\\n")
                f.write("\\cmidrule(r){2-4} \\cmidrule(r){5-7}\n")
                f.write("Baza danych & CPU DB & RAM DB [MiB] & We/Wy DB [MiB] & CPU DB & RAM DB [MiB] & We/Wy DB [MiB] \\\\\n")
                f.write("\\midrule\n")
                for _, row in res_df.iterrows():
                    f.write(f"{row['system']} & {row['insert_cpu']} & {row['insert_mem']} & {row['insert_io']} & "
                            f"{row['read_cpu']} & {row['read_mem']} & {row['read_io']} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}%\n")
                f.write("}\n")
                f.write(f"\\caption{{Porównanie zużycia zasobów (średnie CPU bazy, RAM bazy oraz skumulowane We/Wy odczyt/zapis) dla operacji CRUD (Rekord 128 B, środowisko {host_sys.upper()}, 4 CPU, 8 procesów klienckich, *dla LevelDB/RocksDB 1 proces)}}\n")
                f.write(f"\\label{{tab:porownanie_zasobow_{host_sys}}}\n")
                f.write("\\end{table}\n")
            print(f"Generated comparative resource table: {res_tex_path}")
            
    # Write skipped resource plots to LaTeX table
    tex_skipped_path = os.path.join(os.path.dirname(__file__), "..", "magisterka", "tabelka_zasobow_skipped.tex")
    try:
        with open(tex_skipped_path, "w", encoding="utf-8") as f:
            f.write("% Tabela wygenerowana automatycznie przez skrypt plots/process_results.py\n")
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\resizebox{\\textwidth}{!}{%\n")
            f.write("\\begin{tabular}{llcccc}\n")
            f.write("\\toprule\n")
            f.write("Baza danych & System operacyjny & Rozmiar ładunku & Typ obciążenia & Limit CPU & Próbki \\\\\n")
            f.write("\\midrule\n")
            if skipped_resource_plots:
                # Sort skipped_resource_plots for readability
                sorted_skipped = sorted(skipped_resource_plots, key=lambda x: (x["system"], x["host"], x["payload"], x["iterations"]))
                for item in sorted_skipped:
                    f.write(f"{item['system']} & {item['host']} & {item['payload']} & {item['iterations']} & {item['cpus']} CPU & {item['samples']} \\\\\n")
            else:
                f.write("\\multicolumn{6}{c}{Wszystkie konfiguracje spełniły kryterium minimalnej liczby próbek ($\\ge 30$)} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}%\n")
            f.write("}\n")
            f.write("\\caption{Konfiguracje testowe niespełniające kryterium minimalnej liczby 30 próbek dla wykresów zużycia zasobów}\n")
            f.write("\\label{tab:resource_samples_skipped}\n")
            f.write("\\end{table}\n")
        print(f"Generated global LaTeX skipped resource table: {tex_skipped_path}")
    except Exception as e:
        print(f"Warning: Failed to write skipped resource plots table: {e}")

    # Generate the final ranking table based on Ubuntu (native Linux) results
    generate_ranking_table()

    print(f"\nAll plots and tables generated successfully. Check output folder: {OUTPUT_DIR}")

def generate_ranking_table():
    print("\nCalculating final ranking and scores based on Ubuntu (native Linux) results...")
    csv_path = os.path.join(os.path.dirname(__file__), "aggregated_results_ubuntu.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping ranking table generation.")
        return
        
    df = pd.read_csv(csv_path)
    
    systems = ['redis', 'memcached', 'leveldb', 'rocksdb', 'couchbase', 'tarantool', 'foundationdb', 'berkeleydb']
    sys_display = {
        'redis': 'Redis',
        'memcached': 'Memcached',
        'leveldb': 'LevelDB*',
        'rocksdb': 'RocksDB*',
        'couchbase': 'Couchbase',
        'tarantool': 'Tarantool',
        'foundationdb': 'FoundationDB',
        'berkeleydb': 'BerkeleyDB'
    }
    
    static_scores = {
        'foundationdb': 1500,
        'tarantool': 1450,
        'redis': 1400,
        'couchbase': 1250,
        'berkeleydb': 950,
        'rocksdb': 600,
        'memcached': 350,
        'leveldb': 350
    }
    
    functional_scores = {
        'redis': 700,
        'tarantool': 700,
        'couchbase': 700,
        'memcached': 350,
        'leveldb': 350,
        'rocksdb': 350,
        'berkeleydb': 350,
        'foundationdb': 300,
    }
    
    points_scale = [14, 12, 10, 8, 6, 4, 2, 0]
    perf_scores = {sys: 0 for sys in systems}
    case_count = 0
    
    df_crud = df[df['category'].isin(['insert', 'read', 'update', 'delete'])]
    groups = df_crud.groupby(['payload_size', 'category', 'iterations', 'max_cpus', 'workers'])
    
    for name, group in groups:
        case_count += 1
        sys_throughput = {}
        for sys in systems:
            sys_df = group[group['system'] == sys]
            if not sys_df.empty:
                sys_throughput[sys] = sys_df['throughput_ops_sec'].iloc[0]
            else:
                sys_throughput[sys] = 0.0
                
        sorted_sys = sorted(systems, key=lambda s: sys_throughput[s], reverse=True)
        
        for rank, sys in enumerate(sorted_sys):
            points = points_scale[rank] if rank < len(points_scale) else 0
            if sys_throughput[sys] == 0.0:
                points = 0
            perf_scores[sys] += points
            
    print(f"Total ranked CRUD cases: {case_count}")
    
    rows = []
    for sys in systems:
        static = static_scores[sys]
        func = functional_scores[sys]
        perf = perf_scores[sys]
        total = static + func + perf
        pct = (total / 5560.0) * 100.0
        rows.append({
            'system_id': sys,
            'system': sys_display[sys],
            'static': static,
            'func': func,
            'perf': perf,
            'total': total,
            'pct': f"{pct:.1f}\\%"
        })
        
    rows = sorted(rows, key=lambda x: x['total'], reverse=True)
    
    ranking_tex_path = os.path.join(os.path.dirname(__file__), "..", "magisterka", "tabelka_ranking_koncowy.tex")
    with open(ranking_tex_path, "w", encoding="utf-8") as f:
        f.write("% Tabela wygenerowana automatycznie\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{ccrccrc}\n")
        f.write("\\toprule\n")
        f.write("Pozycja & Baza danych & Cechy statyczne & Funkcjonalność & Wydajność & Suma punktów & \\% maks. oceny \\\\\n")
        f.write("\\midrule\n")
        for idx, row in enumerate(rows):
            f.write(f"{idx+1} & {row['system']} & {row['static']} & {row['func']} & {row['perf']} & {row['total']} & {row['pct']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\caption{Końcowy ranking wielokryterialny badanych systemów bazodanowych na podstawie wyników w środowisku Ubuntu}\n")
        f.write("\\label{tab:ranking_koncowy}\n")
        f.write("\\end{table}\n")
    print(f"Generated final ranking table: {ranking_tex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates benchmark results and generates plots.")
    parser.add_argument("action", choices=["aggregate", "plot", "all"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "aggregate":
        aggregate_results()
    elif args.action == "plot":
        generate_plots()
    elif args.action == "all":
        aggregate_results()
        generate_plots()
