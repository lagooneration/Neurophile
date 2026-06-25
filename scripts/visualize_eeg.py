"""
scripts/visualize_eeg.py

Visualizes EEG sensor locations and a topographic map (head dipole diagram)
for a specific subject in the OpenNeuro BIDS dataset.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import mne
    from mne_bids import BIDSPath, read_raw_bids
except ImportError:
    print("Please install mne-bids: pip install mne-bids matplotlib")
    import sys
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True, help="Path to the downloaded OpenNeuro dataset")
    parser.add_argument("--subject", type=str, default="001", help="Subject ID")
    args = parser.parse_args()

    print(f"Loading EEG data for Subject {args.subject}...")
    # Bypass mne_bids.match() which often fails on Windows PowerShell backslashes
    eeg_dir = args.bids_root / f"sub-{args.subject}" / "eeg"
    set_files = list(eeg_dir.glob("*_eeg.set"))
    
    if not set_files:
        print(f"No .set EEG data found for subject {args.subject} in {eeg_dir}")
        return

    # Use the first matched .set file
    bids_path = BIDSPath(
        subject=args.subject,
        task="AttendedSpeakerParadigmOwnName",
        datatype="eeg",
        suffix="eeg",
        extension=".set",
        root=args.bids_root
    )
    
    # We update the path directly instead of using match()
    bids_path.update(check=False)
    
    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()

    # The BIDS loader automatically parses the electrodes.tsv file to populate raw.info
    # Let's try to set a standard 10-20 montage as a fallback just in case the TSV is missing coordinates
    try:
        if raw.info['dig'] is None or len(raw.info['dig']) == 0:
            print("No sensor locations found in BIDS metadata. Applying standard 10-20 montage...")
            raw.set_montage("standard_1020", match_case=False)
    except Exception as e:
        print(f"Warning: Could not set fallback montage: {e}")

    # --- Plot 1: Sensor positions on the head ---
    print("Generating Head Sensor Map...")
    fig1 = raw.plot_sensors(show_names=True, show=False)
    if fig1 is not None:
        fig1.suptitle(f"Subject {args.subject} - EEG Sensor Dipole Locations")

    # --- Plot 2: Topographic Map (Voltage Distribution) ---
    print("Generating Topographic Map (Brainwave distribution at t=10.0s)...")
    # We grab the raw brainwaves exactly 10 seconds into the recording to plot the topomap
    t_idx = raw.time_as_index(10.0)[0]
    data = raw.get_data(picks="eeg")[:, t_idx]
    
    fig2, ax = plt.subplots(figsize=(6, 5))
    try:
        from mne.viz import plot_topomap
        # Plot the heat map of the voltage across the scalp!
        plot_topomap(data, raw.info, axes=ax, show=False, cmap='RdBu_r')
        fig2.suptitle(f"Subject {args.subject} - Topographic Dipole Map (t=10.0s)")
    except Exception as e:
        print(f"Could not plot topomap: {e}")

    # --- Plot 3: Power Spectral Density (PSD) ---
    print("Generating Power Spectral Density (PSD) Map...")
    try:
        # Compute PSD using Welch's method (limit to 50 Hz to focus on Alpha/Beta/Gamma bands)
        fig3 = raw.compute_psd(fmax=50).plot(show=False)
        fig3.suptitle(f"Subject {args.subject} - Power Spectral Density (PSD)")
    except Exception as e:
        print(f"Could not compute PSD: {e}")

    # --- Plot 4: Event-Related Potential (ERP / P300 / N300) ---
    print("Extracting experimental events to generate ERP...")
    try:
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        # The auditory stimulus trigger is labeled 'beep'
        if 'beep' in event_dict:
            beep_id = event_dict['beep']
            
            # We apply a quick bandpass filter (1-30 Hz) to mathematically clean the data for pure ERP viewing
            print("Applying 1-30 Hz bandpass filter to reveal auditory spikes...")
            raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=30.0, verbose=False)
            
            # Epoch the data: from 200ms BEFORE the beep, to 800ms AFTER the beep
            epochs = mne.Epochs(
                raw_filtered, 
                events, 
                event_id=beep_id, 
                tmin=-0.2, 
                tmax=0.8, 
                baseline=(-0.2, 0),
                preload=True,
                verbose=False
            )
            
            # Mathematically average the hundreds of epochs together to compute the Evoked Response
            evoked = epochs.average()
            print(f"Computed Evoked Response across {len(epochs)} 'beep' auditory trials!")
            
            # Plot the ERP butterfly plot
            fig4 = evoked.plot(spatial_colors=True, show=False)
            fig4.suptitle(f"ERP Butterfly Plot (N100/P300 Response) - Subject {args.subject}")
            
            # --- Inject Latency Markers ---
            # MNE puts the main graph in the first axes object
            ax = fig4.axes[0]
            
            # N100 Marker (approx 100ms)
            ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
            ax.text(0.1, ax.get_ylim()[1]*0.8, ' N100', color='red', fontsize=10, fontweight='bold')
            
            # P200 Marker (approx 200ms)
            ax.axvline(x=0.2, color='blue', linestyle='--', alpha=0.7)
            ax.text(0.2, ax.get_ylim()[1]*0.8, ' P200', color='blue', fontsize=10, fontweight='bold')
            
            # P300/N300 Marker (approx 300ms)
            ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7)
            ax.text(0.3, ax.get_ylim()[1]*0.8, ' P300/N300', color='green', fontsize=10, fontweight='bold')
            
            # Plot the topomap of the ERP at specific time latencies (e.g., 100ms, 300ms)
            fig5 = evoked.plot_topomap(times=[0.1, 0.3, 0.5], show=False)
            fig5.suptitle("ERP Voltage Distribution at 100ms, 300ms, and 500ms post-stimulus")
            
        else:
            print("Could not find 'beep' events to generate ERP. Found events:", event_dict)
    except Exception as e:
        print(f"Failed to generate ERP: {e}")

    print("Displaying plots! (Close the popup windows to exit the script)")
    plt.show()

if __name__ == "__main__":
    main()
