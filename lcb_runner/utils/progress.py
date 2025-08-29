import json

from tqdm import tqdm


class TeeTqdm(tqdm):
    def __init__(self, *args, progress_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_file = progress_file
        self.write_progress_to_file()

    def update(self, n=1):
        super().update(n)
        self.write_progress_to_file()

    def write_progress_to_file(self):
        if self.progress_file:
            progress_dict = self.format_dict
            with open(self.progress_file, "w") as f:
                json.dump(progress_dict, f)