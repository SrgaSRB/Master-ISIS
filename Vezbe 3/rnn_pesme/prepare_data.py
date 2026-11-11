"""Prvi korak: učitavanje pesama i pretvaranje znakova u brojeve i nazad."""

from pathlib import Path
import sys


UNK_TOKEN = "<UNK>"  # Posebna oznaka za znak koji ne postoji u rečniku.


def load_songs(data_dir: Path) -> list[str]:
    """Učitaj .txt fajlove kao UTF-8 i pretvori tekstove u mala slova."""
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"Nema .txt fajlova u folderu: {data_dir}")

    songs = []
    for path in paths:
        text = path.read_text(encoding="utf-8-sig").lower()
        if text.strip():
            songs.append(text)

    if not songs:
        raise ValueError("Fajlovi sa pesmama su prazni ili sadrže samo razmake.")

    return songs


def build_vocabulary(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Napravi dva rečnika od znakova koji se pojavljuju u tekstovima."""
    # set uklanja ponavljanja, a sorted daje isti redosled za isti skup znakova.
    characters = sorted(set("".join(texts)))
    if not characters:
        raise ValueError("Za pravljenje rečnika potreban je neprazan tekst.")

    # Broj 0 čuvamo za nepoznate znakove; poznati znakovi počinju od 1.
    char_to_idx = {UNK_TOKEN: 0}
    for index, character in enumerate(characters, start=1):
        char_to_idx[character] = index

    # Obrni mapiranje: npr. ako je 'a' -> 3, ovde dobijamo 3 -> 'a'.
    idx_to_char = {index: character for character, index in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(text: str, char_to_idx: dict[str, int]) -> list[int]:
    """Zameni svaki znak njegovim brojem; nepoznat znak dobija broj 0."""
    unknown_id = char_to_idx[UNK_TOKEN]
    return [char_to_idx.get(character, unknown_id) for character in text]


def decode(ids: list[int], idx_to_char: dict[int, str]) -> str:
    """Spoji znakove za date brojeve. Nepoznat znak se prikazuje kao <UNK>."""
    return "".join(idx_to_char[index] for index in ids)


def main() -> None:
    # Omogući ispis naših slova i kada Windows preusmeri izlaz u fajl ili alat.
    sys.stdout.reconfigure(encoding="utf-8")

    # Putanja se računa u odnosu na ovaj fajl, nezavisno od mesta pokretanja.
    data_dir = Path(__file__).resolve().parent / "data"
    songs = load_songs(data_dir)

    # Sada koristimo sve probne pesme. Kada dodamo validaciju, rečnik ćemo
    # praviti samo od trening pesama, a isti rečnik koristiti i za validaciju.
    char_to_idx, idx_to_char = build_vocabulary(songs)

    print(f"Broj učitanih pesama: {len(songs)}")
    print(f"Veličina rečnika (uključujući <UNK>): {len(char_to_idx)}")
    print("\nZnak -> broj:")
    for character, index in char_to_idx.items():
        # repr prikazuje novi red kao '\n', tako da se i on vidi u ispisu.
        print(f"  {character!r:8} -> {index}")

    # Prvi stih je iz učitanih podataka, pa su svi njegovi znakovi poznati.
    example = songs[0].splitlines()[0]
    numbers = encode(example, char_to_idx)
    restored = decode(numbers, idx_to_char)

    print(f"\nPočetni tekst: {example!r}")
    print(f"Brojevi:       {numbers}")
    print(f"Vraćeni tekst: {restored!r}")

    # Proveravamo i cele pesme, uključujući razmake i prelome redova.
    for song in songs:
        assert decode(encode(song, char_to_idx), idx_to_char) == song
    print("\nProvera uspešna: svaka učitana pesma vraćena je bez gubitaka.")
    print(f"Nepoznat znak: broj 0 -> {decode([0], idx_to_char)}")


if __name__ == "__main__":
    main()
