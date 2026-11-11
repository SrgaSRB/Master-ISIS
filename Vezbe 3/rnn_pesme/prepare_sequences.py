"""Drugi korak: pravljenje ulaza i očekivanih izlaza za treniranje."""

from pathlib import Path
import sys

from prepare_data import build_vocabulary, decode, encode, load_songs


def create_training_examples(
    songs: list[str],
    char_to_idx: dict[str, int],
    sequence_length: int = 5,
) -> tuple[list[list[int]], list[int]]:
    """Za svaku pesmu napravi parove: prethodni znakovi -> sledeći znak."""
    if sequence_length <= 0:
        raise ValueError("Dužina ulaza mora biti veća od nule.")

    inputs = []
    targets = []

    # Svaku pesmu obrađujemo posebno: prozor ne prelazi u sledeću pesmu.
    for song in songs:
        numbers = encode(song, char_to_idx)

        # Za N znakova i prozor dužine L postoji N - L primera ako je N > L.
        # Kraće pesme nemaju dovoljno znakova za ulaz i sledeći znak.
        for start in range(len(numbers) - sequence_length):
            end = start + sequence_length

            # [start:end] uzima znakove od start do end, bez znaka na end.
            inputs.append(numbers[start:end])
            targets.append(numbers[end])  # Upravo taj sledeći znak je cilj.

    return inputs, targets

def save_training_examples(inputs: list[list[int]], targets: list[int], file_path: Path) -> None:
    """Sačuvaj primere u fajl u formatu CSV: ulaz1,ulaz2,...,ulazN,cilj."""
    with open(file_path, "w", encoding="utf-8") as f:
        for input_ids, target_id in zip(inputs, targets):
            line = ",".join(map(str, input_ids)) + f",{target_id}\n"
            f.write(line)
    

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    data_dir = Path(__file__).resolve().parent / "data"
    songs = load_songs(data_dir)
    char_to_idx, idx_to_char = build_vocabulary(songs)

    sequence_length = 5  # Za početak 5 znakova; kasnije možeš promeniti na 60.
    inputs, targets = create_training_examples(songs, char_to_idx, sequence_length)
    save_training_examples(inputs, targets, data_dir / "training_examples.csv")

    print(f"Broj učitanih pesama: {len(songs)}")
    print(f"Dužina svakog ulaza: {sequence_length}")
    print(f"Broj primera: {len(inputs)}")

    if not inputs:
        print(f"Za bar jedan primer potrebna je pesma sa najmanje {sequence_length + 1} znakova.")
        return

    print("\nPrvih 10 primera kao tekst (ulaz -> cilj):")
    for input_ids, target_id in zip(inputs[:10], targets[:10]):
        input_text = decode(input_ids, idx_to_char)
        # decode prima listu, pa i jedan broj stavljamo u listu.
        target_text = decode([target_id], idx_to_char)
        print(f"  {input_text!r} -> {target_text!r}")

    print("\nPrvi primer kao brojevi:")
    print(f"  Ulaz: {inputs[0]}")
    print(f"  Cilj: {targets[0]}")
    print("\nMreža će kasnije iz ovih ulaza učiti da predviđa ciljeve.")


if __name__ == "__main__":
    main()
