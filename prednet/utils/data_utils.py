import hickle as hkl
import numpy as np

from torch.utils.data import Dataset

class kittidata(Dataset):
    def __init__(self, data_path, source_path, nt, N_seq, sequence_start_mode='all'):
        datafile = open(data_path, 'r')
        self.X = hkl.load(datafile)
        datafile.close()
        sourcefile = open(source_path, 'r')
        self.sources = hkl.load(sourcefile)
        sourcefile.close()
        self.nt = nt
        self.N_seq = N_seq
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        self.X = np.transpose(self.X, (0, 3, 1, 2))

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts
        self.possible_starts_N_seq = self.possible_starts[:N_seq]

    def __len__(self):
        return len(self.possible_starts_N_seq)

    def __getitem__(self, idx):
        x = self.preprocess(self.X[self.possible_starts_N_seq[idx]:self.possible_starts_N_seq[idx]+self.nt])
        return x

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def shuffle(self):
        self.possible_starts_N_seq = np.random.permutation(self.possible_starts)[:self.N_seq]
