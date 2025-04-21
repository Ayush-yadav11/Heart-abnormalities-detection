import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # Load data from CSV
    csv_path = "src/python/pulse_rate_dataset.csv"
    try:
        data = pd.read_csv(csv_path)
        if len(data.columns) != 3 or not all(col in data.columns for col in ['time', 'pulse_rate', 'spo2']):
            print(f"Error: CSV file must have columns: time, pulse_rate, spo2")
            return
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please ensure the data file exists.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Convert all columns to numeric type
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()  # Remove any rows with invalid data
    if len(data) == 0:
        print("Error: No valid data found in CSV file")
        return
    
    # Initialize data storage with larger window for better trend visibility
    max_points = 200  # Increased buffer size for longer history
    times = deque(maxlen=max_points)
    pulse_rates = deque(maxlen=max_points)
    spo2_values = deque(maxlen=max_points)
    
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Real-time Vital Signs Monitoring')
    
    # Initialize empty lines
    line1, = ax1.plot([], [], 'b-', label='Pulse Rate')
    line2, = ax2.plot([], [], 'r-', label='SpO2')
    
    # Set up axes
    ax1.set_ylim(40, 200)
    ax1.set_ylabel('BPM')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_ylim(80, 100)
    ax2.set_ylabel('%')
    ax2.grid(True)
    ax2.legend()
    
    # Add threshold lines for abnormal heart rates
    ax1.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Bradycardia')
    ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Tachycardia')
    
    def update(frame):
        if frame >= len(data):
            return line1, line2
        
        # Get current values
        current_time = data['time'].iloc[frame]
        pulse_rate = data['pulse_rate'].iloc[frame]
        spo2 = data['spo2'].iloc[frame]
        
        # Update data
        times.append(current_time)
        pulse_rates.append(pulse_rate)
        spo2_values.append(spo2)
        
        # Update plot data
        line1.set_data(list(times), list(pulse_rates))
        line2.set_data(list(times), list(spo2_values))
        
        # Adjust x axis limits
        for ax in [ax1, ax2]:
            ax.set_xlim(max(0, current_time - 10), current_time + 3)
        
        return line1, line2
    
    print("Starting vital signs monitoring...")
    try:
        ani = FuncAnimation(fig, update, frames=len(data), interval=50, blit=True, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping analysis...")
    print("Analysis complete!")

if __name__ == "__main__":
    main()