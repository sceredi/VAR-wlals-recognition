from handcrafted.app.dataset.dataset import Dataset


class Launcher:
    def start(self) -> None:
        print("Oui oui")
        print(len(self._load_data().videos))

    def _load_data(self) -> Dataset:
        return Dataset("data/WLASL_v0.3.json")
