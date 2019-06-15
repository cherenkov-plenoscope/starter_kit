import os
import numpy as np
import plenopy as pl


class SimpleIoRun():
    def __init__(self, path):
        """
        Parameters
        ----------
        path        The path to the directory representing the run.
        """
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError(self.path)
        self.event_numbers = self._event_numbers_in_run()
        self.number_events = self.event_numbers.shape[0]
        self.header = pl.corsika.RunHeader(
            os.path.join(path, 'corsika_run_header.bin'))

    def _event_numbers_in_run(self):
        return pl.tools.acp_format.all_folders_with_digit_names_in_path(
            self.path)

    def __getitem__(self, index):
        """
        Returns the index-th event of this run.

        Parameters
        ----------
        index       The index of the event to be returned. (starting at 0).
        """
        try:
            event_number = self.event_numbers[index]
        except(IndexError):
            raise StopIteration
        event_path = os.path.join(self.path, str(event_number))
        return SimpleIoEvent(event_path)

    def __len__(self):
        return self.number_events

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += str(self.number_events) + ' events)'
        out += ')'
        return out


class SimpleIoEvent():
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError(self.path)
        self.header = pl.corsika.EventHeader(
            os.path.join(path, "corsika_event_header.bin"))

        self.cherenkov_photon_bunches = pl.corsika.PhotonBunches(
            os.path.join(path, "air_shower_photon_bunches.bin"))

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += 'number '+str(self.header.number)
        out += ')'
        return out
