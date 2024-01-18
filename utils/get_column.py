from dataclasses import dataclass

@dataclass
class GetColumn():
    column_dict: dict

    @property
    def all(self):
        return list(self.column_dict.values())

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return [self.column_dict[key] for key in index]
        return self.column_dict[index]
    