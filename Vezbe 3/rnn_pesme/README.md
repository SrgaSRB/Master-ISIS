# RNN pesme — priprema podataka

Za početak kodiramo **znakove**: svako slovo, razmak, znak interpunkcije i novi
red dobijaju svoj ceo broj. To odgovara priloženom planu i traži mnogo manji
rečnik nego kodiranje celih reči. Mreža će kasnije predviđati sledeći znak.

U ovoj fazi još nema treniranja ni generisanja pesama. Brojevi su samo oznake
znakova: veći broj ne znači da je znak važniji ili sličniji nekom drugom znaku.

## Pokretanje

Za prva dva koraka potreban je Python 3.9 ili noviji, bez dodatnih biblioteka.
Iz foldera projekta pokreni:

```powershell
python prepare_data.py
```

Program učitava `.txt` fajlove iz foldera `data`, pravi rečnik, ispisuje jedan
stih kao brojeve i vraća ga u tekst. Zatim proverava isti postupak za cele pesme.
`data/pesma_01.txt` je izmišljen probni tekst za vežbu. Svoje tekstove dodaj kao
UTF-8 `.txt` fajlove, jednu pesmu po fajlu.

## Kako radi

1. `load_songs` učitava pesme i pretvara ih u mala slova. Razmaci, interpunkcija
   i prelomi redova ostaju sačuvani.
2. `build_vocabulary` izdvaja različite znakove, sortira ih i pravi dva rečnika:
   `char_to_idx` (znak → broj) i `idx_to_char` (broj → znak).
3. `encode` prolazi kroz tekst i svaki znak zamenjuje brojem.
4. `decode` pronalazi znak za svaki broj i spaja znakove u tekst.

Mali primer koji možeš pokrenuti u Pythonu:

```python
from prepare_data import build_vocabulary, encode, decode

char_to_idx, idx_to_char = build_vocabulary(["ana"])

print(char_to_idx)                        # {'<UNK>': 0, 'a': 1, 'n': 2}
print(idx_to_char)                        # {0: '<UNK>', 1: 'a', 2: 'n'}
print(encode("ana", char_to_idx))         # [1, 2, 1]
print(decode([1, 2, 1], idx_to_char))      # ana

text = "ana"
assert decode(encode(text, char_to_idx), idx_to_char) == text
```

`encode` i `decode` sami ne menjaju velika i mala slova. Kod učitanih pesama
provera se odnosi na tekst nakon pretvaranja u mala slova. Kada kasnije unosiš
početak pesme, koristi `encode(pocetni_tekst.lower(), char_to_idx)`.

Broj `0` je rezervisan za `<UNK>` — nepoznat znak. Na primer, uz gornji rečnik:

```python
print(encode("ana!", char_to_idx))        # [1, 2, 1, 0]
print(decode([1, 2, 1, 0], idx_to_char))   # ana<UNK>
```

Nepoznat znak se ne može vratiti u original, jer njegov identitet nije sačuvan.
Za poznate znakove važi `decode(encode(tekst)) == tekst`, uz prosleđene rečnike.
`decode` očekuje samo brojeve koji postoje u rečniku; za nepostojeći broj Python
prijavljuje `KeyError`.

Sačuvaj i koristi isti par rečnika za oba smera: dodavanje novih znakova i ponovno
pravljenje rečnika može promeniti brojeve. Kada dodamo podelu na trening i
validaciju, rečnik ćemo graditi samo iz trening pesama.

## Drugi korak: primeri za trening

Pokreni novi fajl:

```powershell
python prepare_sequences.py
```

Funkcija `create_training_examples` koristi postojeći rečnik i pomera prozor
kroz svaku pesmu za po jedan znak. Pet prethodnih znakova čini **ulaz**, a znak
odmah iza njih je **cilj** koji će mreža učiti da predvidi. Na tekstu `volim te`:

```text
Ulaz      Cilj
'volim'   ' '
'olim '   't'
'lim t'   'e'
```

Funkcija vraća dve liste: `inputs` sadrži ulaze kao liste brojeva, a `targets`
sadrži po jedan broj za svaki cilj. `inputs[0]` i `targets[0]` čine prvi par.
Program prikazuje prvih deset parova kao tekst, pa prvi par kao brojeve.

Za pesmu dužine `N` i prozor dužine `L` dobijaš `N - L` primera ako je `N > L`.
Pesme sa `L` ili manje znakova ne daju nijedan primer. Prozori mogu obuhvatiti
novi red unutar pesme, ali nikada ne prelaze iz jedne pesme u drugu.

U funkciji `main` promeni `sequence_length = 5` u `60` kada želiš duži kontekst.
Za ovaj korak i dalje je dovoljan običan Python.

## Treći korak: Dataset i DataLoader

Od ovog koraka koristimo PyTorch. Projekat proveravamo na Pythonu 3.13,
uz PyTorch 2.15.0 (CPU) i NumPy 2.5.4; verzije su navedene u `requirements.txt`.
Biblioteke su odvojene u lokalnom okruženju `.venv`. Za postavljanje projekta
na drugom računaru, iz foldera projekta pokreni:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Koristimo CPU verziju PyTorch-a, pa za ovu vežbu nije potrebna posebna grafička
karta. Kada je okruženje postavljeno, pokreni:

```powershell
.\.venv\Scripts\python.exe dataset.py
```

`SongDataset` koristi funkciju iz prethodnog koraka i pretvara njene liste u
**tenzore** — nizove brojeva sa kojima radi PyTorch. Ima tri kratke metode:

- `__init__` priprema ulaze i ciljeve kao cele brojeve (`torch.long`).
- `__len__` vraća broj primera.
- `__getitem__` vraća jedan ulaz i njegov cilj.

`DataLoader` uzima te parove i grupiše ih u **batch-eve**. Sa probnom pesmom,
prozorom od 5 znakova i grupom od najviše 64 primera, prvi batch ima oblike:

```text
Ulazi:   [64, 5]  — 64 primera, svaki sa 5 znakova
Ciljevi: [64]     — po jedan sledeći znak za svaki primer
```

Ukupno ima 98 primera: prvi batch ima 64, a poslednji 34. `shuffle=True` menja
redosled primera pri svakom prolasku, ali čuva vezu ulaza i cilja i redosled
znakova unutar ulaza. Zato se prikazani primeri mogu razlikovati pri pokretanju.
Ako prozor povećaš na 60, ova kratka pesma daje 43 primera i jedan batch oblika
`[43, 60]`, sa ciljevima `[43]`.

Više o ovom interfejsu: [zvanični PyTorch primer za Dataset i DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).

## Šta još ostaje

Posle ova tri mala koraka ostaje **6 koraka** do završetka planirane vežbe:

1. **Podela podataka:** uklanjanje duplih pesama i podela celih pesama na trening
   i validaciju pre pravljenja prozora; rečnik pravimo samo iz trening pesama.
2. **RNN model:** Embedding → RNN → linearni izlaz i provera jednog prolaska.
3. **Treniranje i validacija:** računanje greške, ažuriranje težina i praćenje
   rezultata po epohama.
4. **Provera učenja i čuvanje:** prvo proveriti da mreža može da zapamti mali
   uzorak, pa sačuvati najbolji model, rečnik i podešavanja tokom punog treninga.
5. **Generisanje:** učitati model i nastavljati početni tekst znak po znak,
   uz podešavanje temperature.
6. **Eksperimenti:** uporediti rezultate pri promeni temperature, dužine
   konteksta ili veličine mreže.

Brojimo praktične korake koje ovde radimo, pa se numeracija razlikuje od
prvobitnog plana. Za smislenu validaciju potrebno je više različitih pesama;
trenutni jedan probni tekst služi za proveru koda.
