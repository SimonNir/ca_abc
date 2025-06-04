import numpy as np
import os
import h5py
from bias import GaussianBias

class ABCStorage:
    def __init__(self, run_dir="abc_data", iterations_per_file=50):
        """
        Simplified storage system that:
        - Groups N iterations per file with iteration ranges in filenames
        - Maintains persistent bias/minima/saddles files
        - Enables fast lookup using filename patterns
        """
        self.run_dir = run_dir
        self.iterations_per_file = iterations_per_file
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize persistent files
        self.biases_file = f"{run_dir}/biases.h5"
        self.landmarks_file = f"{run_dir}/landmarks.h5"

        # Initialize empty files if they don't exist
        for fname in [self.biases_file, self.landmarks_file]:
            if not os.path.exists(fname):
                with h5py.File(fname, 'w') as f:
                    pass  # Create empty file

    def _get_iter_filename(self, iteration):
        """Generate filename containing iteration range"""
        start_iter = (iteration // self.iterations_per_file) * self.iterations_per_file
        end_iter = start_iter + self.iterations_per_file - 1
        return f"{self.run_dir}/iters_{start_iter:06d}_{end_iter:06d}.h5"

    def dump_current(self, abc_instance):
        """Dump current ABC data with iteration range in filename"""
        iteration = abc_instance.current_iteration
        filename = self._get_iter_filename(iteration)
        
        # 1. Prepare trajectory data
        history = {
            'positions': np.array(abc_instance.trajectory),
            'biased_energies': np.array(abc_instance.biased_energies),
            'unbiased_energies': np.array(abc_instance.unbiased_energies),
            'biased_forces': np.array(abc_instance.biased_forces),
            'unbiased_forces': np.array(abc_instance.unbiased_forces),
        }

        # 2. Atomic write of trajectory data with checks 
        tmp_file = f"{filename}.tmp"
        try: 
            with h5py.File(tmp_file, 'w') as f:
                # Store iteration range as attributes
                f.attrs['iter_start'] = (iteration // self.iterations_per_file) * self.iterations_per_file
                f.attrs['iter_end'] = f.attrs['iter_start'] + self.iterations_per_file - 1
                f.attrs['iterations_per_file'] = self.iterations_per_file  # For validation

                # Store current iteration data
                iter_group = f.create_group(f"iter_{iteration:06d}")
                for key, value in history.items():
                    if isinstance(value, np.ndarray):
                        iter_group.create_dataset(key, data=value)
                    else:
                        iter_group.attrs[key] = value
                
            os.rename(tmp_file, filename)
        except Exception as e:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            raise RuntimeError(f"Failed to write iteration chunk file: {str(e)}")
        
        # 3. Append to persistent files (biases and landmarks)
        self._append_biases(iteration, abc_instance.bias_list)
        self._append_minima(iteration, abc_instance.minima)
        self._append_saddles(iteration, abc_instance.saddles)
        self._append_iter_periods(iteration, abc_instance.iter_periods)

    def _append_biases(self, iteration, biases):
        """Append the most recent biases to persistent storage"""
        if not biases:
            return
        # Add the last N biases, where N = self.iterations_per_file (or dump_every)
        recent_biases = biases[-self.iterations_per_file:]
        with h5py.File(self.biases_file, 'a') as f:
            iter_group = f.create_group(f"iter_{iteration}")
            for i, bias in enumerate(recent_biases):
                bg = iter_group.create_group(f"bias_{i}")
                bg.create_dataset('center', data=bias.center)
                bg.create_dataset('covariance', data=bias.covariance)
                bg.attrs['height'] = bias.height

    def _append_minima(self, iteration, minima):
        """Append the most recent minima to persistent landmarks file"""
        if not minima:
            return
        recent_minima = minima[-self.iterations_per_file:]
        with h5py.File(self.landmarks_file, 'a') as f:
            if 'minima' not in f:
                f.create_group('minima')
            for i, min_pos in enumerate(recent_minima):
                f['minima'].create_dataset(f"{iteration}_{i}", data=min_pos)

    def _append_saddles(self, iteration, saddles):
        """Append the most recent saddles to persistent landmarks file"""
        if not saddles:
            return
        recent_saddles = saddles[-self.iterations_per_file:]
        with h5py.File(self.landmarks_file, 'a') as f:
            if 'saddles' not in f:
                f.create_group('saddles')
            for i, saddle_pos in enumerate(recent_saddles):
                f['saddles'].create_dataset(f"{iteration}_{i}", data=saddle_pos)

    def _append_iter_periods(self, iteration, iter_periods):
        """Append the most recent iter_periods to persistent landmarks file"""
        if not iter_periods:
            return
        recent_iter_periods = iter_periods[-self.iterations_per_file:]
        with h5py.File(self.landmarks_file, 'a') as f:
            if 'iter_periods' not in f:
                f.create_group('iter_periods')
            for i, iter_period in enumerate(recent_iter_periods):
                f['iter_periods'].create_dataset(f"{iteration}_{i}", data=iter_period)

    def load_iteration(self, iteration):
        """Load data for specific iteration"""

        if not 0 <= iteration <= 999999:
            raise ValueError("Iteration must be between 0 and 999,999")
            
        filename = self._get_iter_filename(iteration)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No data found for iteration {iteration}")
        
        history = {}
        
        # Load trajectory data
        with h5py.File(filename, 'r') as f:
            iter_group = f[f"iter_{iteration}"]
            for key in iter_group:
                if isinstance(iter_group[key], h5py.Dataset):
                    history[key] = iter_group[key][:]
                else:
                    history[key] = iter_group[key].attrs[key]
        
        # Load biases
        biases = []
        with h5py.File(self.biases_file, 'r') as f:
            if f"iter_{iteration}" in f:
                bias_group = f[f"iter_{iteration}"]
                for bias_key in bias_group:
                    bg = bias_group[bias_key]
                    biases.append(GaussianBias(
                        center=bg['center'][:],
                        covariance=bg['covariance'][:],
                        height=bg.attrs['height']
                    ))
        
        # Load landmarks
        minima, saddles = [], []
        with h5py.File(self.landmarks_file, 'r') as f:
            if 'minima' in f:
                minima = [f['minima'][key][:] for key in f['minima'] if str(iteration) in key]
            if 'saddles' in f:
                saddles = [f['saddles'][key][:] for key in f['saddles'] if str(iteration) in key]
        
        return {
            'history': history,
            'biases': biases,
            'minima': minima,
            'saddles': saddles
        }

    def get_all_files(self):
        """Get all iters files in chronological order"""
        files = sorted([f for f in os.listdir(self.run_dir) if f.startswith('iters_')])
        return [os.path.join(self.run_dir, f) for f in files]
    
    # Additional utility methods
    def find_iterations_in_range(self, start_iter, end_iter):
        """Find all iterations within a range"""
        files = []
        current = start_iter
        while current <= end_iter:
            start_iter = (current // self.iterations_per_file) * self.iterations_per_file
            end_iter = start_iter + self.iterations_per_file - 1
            filename = f"iters_{start_iter:06d}_{end_iter:06d}.h5"
            if os.path.exists(os.path.join(self.run_dir, filename)):
                files.append((start_iter, end_iter, filename))
            current = end_iter + 1
        return files

    def get_total_iterations(self):
        """Count total iterations stored"""
        chunks = sorted([f for f in os.listdir(self.run_dir) if f.startswith('iters_')])
        if not chunks:
            return 0
        last_chunk = chunks[-1]
        with h5py.File(os.path.join(self.run_dir, last_chunk), 'r') as f:
            return f.attrs['iter_end'] + 1  # +1 since end is inclusive
        

    def get_most_recent_iter(self):
        """Return the most recent iteration number stored, or None if none exist."""
        files = self.get_all_files()
        if not files:
            return None
        last_file = files[-1]
        with h5py.File(last_file, 'r') as f:
            return f.attrs['iter_end']