import numpy as np
import os
import h5py
from bias import GaussianBias

class ABCStorage:
    def __init__(self, run_dir="abc_data", iterations_per_file=50):
        """
        Storage system that handles all data persistence for ABC simulations.
        - Groups N iterations per file with iteration ranges in filenames
        - Maintains persistent bias/minima/saddles files
        - Provides clean interface for data access
        """
        self.run_dir = run_dir
        self.iterations_per_file = iterations_per_file
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize persistent files
        self.biases_file = os.path.join(run_dir, "biases.h5")
        self.landmarks_file = os.path.join(run_dir, "landmarks.h5")

        # Initialize empty files if they don't exist
        for fname in [self.biases_file, self.landmarks_file]:
            if not os.path.exists(fname):
                with h5py.File(fname, 'w') as f:
                    pass  # Create empty file

    def _get_iter_filename(self, iteration):
        """Generate filename containing iteration range"""
        start_iter = (iteration // self.iterations_per_file) * self.iterations_per_file
        end_iter = start_iter + self.iterations_per_file - 1
        return os.path.join(self.run_dir, f"iters_{start_iter:06d}_{end_iter:06d}.h5")

    def dump_current(self, abc_instance):
        """Dump current ABC data with iteration range in filename"""
        iteration = abc_instance.current_iteration
        filename = self._get_iter_filename(iteration)
        
        # Prepare trajectory data
        history = {
            'positions': np.array(abc_instance.trajectory),
            'biased_energies': np.array(abc_instance.biased_energies),
            'unbiased_energies': np.array(abc_instance.unbiased_energies),
            'biased_forces': np.array(abc_instance.biased_forces),
            'unbiased_forces': np.array(abc_instance.unbiased_forces),
        }

        # Atomic write of trajectory data
        tmp_file = f"{filename}.tmp"
        try: 
            with h5py.File(tmp_file, 'w') as f:
                # Store iteration range as attributes
                f.attrs['iter_start'] = (iteration // self.iterations_per_file) * self.iterations_per_file
                f.attrs['iter_end'] = f.attrs['iter_start'] + self.iterations_per_file - 1
                f.attrs['iterations_per_file'] = self.iterations_per_file

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
        
        # Append to persistent files
        self._append_biases(iteration, abc_instance.bias_list)
        self._append_landmarks(iteration, 'minima', abc_instance.minima)
        self._append_landmarks(iteration, 'saddles', abc_instance.saddles)
        self._append_landmarks(iteration, 'iter_periods', abc_instance.iter_periods)

    def _append_biases(self, iteration, biases):
        """Append biases to persistent storage"""
        if not biases:
            return
            
        recent_biases = biases[-self.iterations_per_file:]
        with h5py.File(self.biases_file, 'a') as f:
            iter_group = f.create_group(f"iter_{iteration}")
            for i, bias in enumerate(recent_biases):
                bg = iter_group.create_group(f"bias_{i}")
                bg.create_dataset('center', data=bias.center)
                bg.create_dataset('covariance', data=bias.covariance)
                bg.attrs['height'] = bias.height

    def _append_landmarks(self, iteration, landmark_type, landmarks):
        """Generic method to append landmarks (minima/saddles/periods)"""
        if not landmarks:
            return
            
        recent_landmarks = landmarks[-self.iterations_per_file:]
        with h5py.File(self.landmarks_file, 'a') as f:
            if landmark_type not in f:
                f.create_group(landmark_type)
            for i, landmark in enumerate(recent_landmarks):
                f[landmark_type].create_dataset(f"{iteration}_{i}", data=landmark)

    def load_iteration(self, iteration):
        """Load complete data for specific iteration"""
        if not 0 <= iteration <= 999999:
            raise ValueError("Iteration must be between 0 and 999,999")
            
        filename = self._get_iter_filename(iteration)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No data found for iteration {iteration}")
        
        # Load trajectory data
        with h5py.File(filename, 'r') as f:
            iter_group = f[f"iter_{iteration:06d}"]
            history = {
                key: iter_group[key][:] if isinstance(iter_group[key], h5py.Dataset) 
                else iter_group.attrs[key]
                for key in iter_group
            }
        
        # Load biases
        biases = []
        with h5py.File(self.biases_file, 'r') as f:
            if f"iter_{iteration}" in f:
                for bias_key in f[f"iter_{iteration}"]:
                    bg = f[f"iter_{iteration}"][bias_key]
                    biases.append(GaussianBias(
                        center=bg['center'][:],
                        covariance=bg['covariance'][:],
                        height=bg.attrs['height']
                    ))
        
        # Load landmarks
        landmarks = {}
        with h5py.File(self.landmarks_file, 'r') as f:
            for landmark_type in ['minima', 'saddles', 'iter_periods']:
                if landmark_type in f:
                    landmarks[landmark_type] = [
                        f[landmark_type][key][:] 
                        for key in f[landmark_type] 
                        if str(iteration) in key
                    ]
        
        return {
            'history': history,
            'biases': biases,
            **landmarks
        }

    def get_all_files(self):
        """Get all iteration files in chronological order"""
        files = sorted(
            f for f in os.listdir(self.run_dir) 
            if f.startswith('iters_') and f.endswith('.h5')
        )
        return [os.path.join(self.run_dir, f) for f in files]
    
    def find_iterations_in_range(self, start_iter, end_iter):
        """Find all iterations within a range"""
        files = []
        current = start_iter
        while current <= end_iter:
            start_file_iter = (current // self.iterations_per_file) * self.iterations_per_file
            end_file_iter = start_file_iter + self.iterations_per_file - 1
            filename = f"iters_{start_file_iter:06d}_{end_file_iter:06d}.h5"
            if os.path.exists(os.path.join(self.run_dir, filename)):
                files.append((start_file_iter, end_file_iter, filename))
            current = end_file_iter + 1
        return files

    def get_total_iterations(self):
        """Count total iterations stored"""
        chunks = self.get_all_files()
        if not chunks:
            return 0
        last_chunk = chunks[-1]
        with h5py.File(last_chunk, 'r') as f:
            return f.attrs['iter_end'] + 1  # +1 since end is inclusive
        
    def get_most_recent_iter(self):
        """Return the most recent iteration number stored, or None if none exist."""
        files = self.get_all_files()
        if not files:
            return None
        last_file = files[-1]
        with h5py.File(last_file, 'r') as f:
            return f.attrs['iter_end']