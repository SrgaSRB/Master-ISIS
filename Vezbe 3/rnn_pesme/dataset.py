"""Treći korak: pretvaranje primera u PyTorch tenzore i grupe za trening."""

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset

from prepare_data import build_vocabulary, decode, load_songs
from prepare_sequences import create_training_examples


class SongDataset(Dataset):
    """Čuva primere i omogućava da uzmemo jedan par: ulaz i cilj."""

    def __init__(
        self,
        songs: list[str],
        char_to_idx: dict[str, int],
        sequence_length: int = 5,
    ) -> None:
        inputs, targets = create_training_examples(songs, char_to_idx, sequence_length)
        if not inputs:
            raise ValueError(
                f"Nema primera: potrebna je pesma sa najmanje {sequence_length + 1} znakova."
            )

        # Tenzor je PyTorch-ov niz brojeva. long čuva cele brojeve (oznake znakova).
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        """Poziva se kada napišemo len(dataset): vraća broj primera."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Poziva se za dataset[index]: vraća jedan ulaz i njegov cilj."""
        return self.inputs[index], self.targets[index]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    data_dir = Path(__file__).resolve().parent / "data"
    songs = load_songs(data_dir)
    char_to_idx, idx_to_char = build_vocabulary(songs)

    sequence_length = 5
    batch_size = 64  # Najviše 64 primera u jednoj grupi (batch-u).
    dataset = SongDataset(songs, char_to_idx, sequence_length)

    # Mešamo redosled primera, ali znakovi unutar svakog ulaza ostaju na mestu.
    # Poslednja grupa može biti manja: nijedan primer se ne odbacuje.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Uzimamo samo prvu grupu za prikaz. Kasnije će trening prolaziti kroz sve.
    inputs, targets = next(iter(loader))
    print(f"Broj primera: {len(dataset)}")
    print(f"Broj grupa: {len(loader)}")
    print(f"Oblik ulaza: {list(inputs.shape)}")
    print(f"Oblik ciljeva: {list(targets.shape)}")
    print(f"Tip brojeva: {inputs.dtype}")

    print("\nPrva tri primera iz grupe (ulaz -> cilj):")
    for input_ids, target_id in zip(inputs[:3], targets[:3]):
        # tolist() vraća običnu Python listu, a item() jedan običan broj.
        input_text = decode(input_ids.tolist(), idx_to_char)
        target_text = decode([target_id.item()], idx_to_char)
        print(f"  {input_text!r} -> {target_text!r}")


if __name__ == "__main__":
    main()
