import numpy as np
import pandas as pd
from io import StringIO

class Read:
    def __init__(self, filename):
        if hasattr(filename, "read"):
            output = filename
        else:
            output = open(filename, 'r')
        self.read_file(output)

    def read_file(self, fileobj):
        """Read Gibbs-Duhem output file
        """
        string = fileobj.readline()  # string should start with the kws
        self.keywords = string.split()
        contents = fileobj.read()
        self.contents = pd.read_table(StringIO(string + contents), sep=r'\s+')

    def find(self, entry_name):
        return np.asarray(self.contents[entry_name])

    def get_keywords(self):
        """Return list of available data columns in the log file."""
        print(", ".join(self.keywords))
