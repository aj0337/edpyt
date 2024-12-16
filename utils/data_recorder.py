from mpi4py import MPI
import h5py

class DMFTDataRecorder:
    """
    A data recorder specifically designed for saving DMFT step data to an HDF5 file.

    Attributes:
        filename (str): The path and filename of the HDF5 file.
        store_last_n (int): The number of last DMFT iterations to store for delta, sigma, and gfloc.
    """

    def __init__(self, filename, store_last_n):
        """
        Initialize the DMFTDataRecorder with a specified filename and storage limit for the last n iterations.

        Parameters:
            filename (str): The path and filename for storing the DMFT data.
            store_last_n (int): Number of last DMFT iterations to store for delta, sigma, and gfloc.
        """
        self.filename = filename
        self.store_last_n = store_last_n
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Only initialize the HDF5 file structure if on rank 0
        if self.rank == 0:
            with h5py.File(self.filename, "w") as f:
                f.create_group("last_n_iterations")  # For delta, sigma, gfloc (last n only)
                f.create_group("all_bath_parameters")  # For vk, ek (all iterations)

    def save_iteration_data(self, iter_num, delta, sigma, gfloc, gfimp):
        """
        Save DMFT iteration data to the HDF5 file, retaining all bath parameters
        for each iteration and limiting delta, sigma, and gfloc storage to the last n iterations.

        Parameters:
            iter_num (int): The current DMFT iteration number.
            delta (np.ndarray): The delta data for the current iteration.
            sigma (np.ndarray): The sigma data for the current iteration.
            gfloc (np.ndarray): The gfloc data for the current iteration.
            gfimp (list): A list of gfimp objects from which bath parameters are extracted.
        """
        # Only rank 0 will write to the HDF5 file
        if self.rank == 0:
            with h5py.File(self.filename, "a") as f:
                # Store bath parameters vk and ek for each gf in gfimp
                bath_grp = f["all_bath_parameters"]
                iter_bath_grp = bath_grp.create_group(f"iteration_{iter_num}")

                for i, gf in enumerate(gfimp):
                    iter_bath_grp.create_dataset(f"vk_{i}", data=gf.Delta.vk)
                    iter_bath_grp.create_dataset(f"ek_{i}", data=gf.Delta.ek)

                # Store only last n iterations for delta, sigma, gfloc
                iter_grp = f["last_n_iterations"]
                if len(iter_grp.keys()) >= self.store_last_n:
                    oldest_iter = sorted(iter_grp.keys())[0]
                    del iter_grp[oldest_iter]

                grp = iter_grp.create_group(f"iteration_{iter_num}")
                grp.create_dataset("delta", data=delta)
                grp.create_dataset("sigma", data=sigma)
                grp.create_dataset("gfloc", data=gfloc)
