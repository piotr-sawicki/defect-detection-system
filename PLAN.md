# Project Plan

> Status: **implementing**
> Created: 2026-05-11
> Last Updated: 2026-05-11

## Objective

Przygotowanie 15-minutowej prezentacji: „System detekcji defektów oparty na analizie obrazów z wykorzystaniem metod sztucznej inteligencji"
Prowadzący: Piotr Sawicki, Power&Controls, GETC

---

## Struktura prezentacji (15 minut)

### Slajd 1 — Tytuł (30 s)

**Tytuł:** System detekcji defektów oparty na analizie obrazów z wykorzystaniem metod sztucznej inteligencji

**Podtytuł:** Faster R-CNN vs YOLOv8 na zbiorze NEU-DET
Piotr Sawicki | Power&Controls | GETC | 2026

---

### Slajd 2 — Agenda (30 s)

1. Problem i motywacja
2. Pipeline budowy systemu
3. Zbiór danych NEU-DET
4. Modele: Faster R-CNN i YOLOv8
5. Wyniki i porównanie
6. Narzędzie inżynierskie (demo)
7. Fine-tuning i dane publiczne

---

### Slajd 3 — Problem i motywacja (1 min)

**Tytuł:** Dlaczego automatyczna detekcja defektów?

**Treść:**
- Kontrola jakości w przemyśle: stale walcowane, odlewy, spoiny
- Inspekcja ręczna: wolna, kosztowna, podatna na błędy człowieka
- Defekty na stalowych taśmach walcowanych → złom, reklamacje klientów
- Cel: system real-time, powtarzalny, skalowalny

**Wizualizacja:** przykładowe zdjęcia defektów (crazing, patches, scratches) z folderu `demo/`

---

### Slajd 4 — Pipeline budowy systemu (1.5 min)

**Tytuł:** Pełny pipeline — od danych do narzędzia

```
Dane surowe (obrazy + anotacje)
        ↓
Przygotowanie zbioru (konwersja formatów, podział train/val)
        ↓
Trening modeli deep learning (GPU)
        ↓
Ewaluacja i porównanie modeli
        ↓
Integracja w narzędziu inżynierskim (FastAPI + UI)
```

**Kluczowe punkty:**
- Każdy etap jest skryptowalny i powtarzalny
- Modele wymienne dzięki wzorcowi Strategy
- Lazy-loading — brak opóźnienia przy starcie serwera

---

### Slajd 5 — Zbiór danych NEU-DET (1.5 min)

**Tytuł:** Zbiór danych — NEU Surface Defect Database

| Klasa defektu    | Liczba obrazów |
|------------------|---------------|
| crazing          | 300           |
| inclusion        | 300           |
| patches          | 300           |
| pitted_surface   | 300           |
| rolled-in_scale  | 300           |
| scratches        | 300           |
| **Razem**        | **1800**      |

- Skala szarości, 200×200 px, gorąco walcowane taśmy stalowe
- Anotacje w formacie Pascal VOC XML
- Konwersja do formatu YOLO: `scripts/prepare_dataset.py`
- Podział: ~80% trening / 20% walidacja (360 obrazów walidacyjnych)

---

### Slajd 6 — Architektura modeli (2 min)

**Tytuł:** Dwa podejścia do detekcji obiektów

**Faster R-CNN (ResNet50)**
- Architektura dwuetapowa: Region Proposal Network → klasyfikacja i regresja
- Trenowany od podstaw na NEU-DET
- Wyższa dokładność, wolniejszy inference
- Implementacja: PyTorch / torchvision

**YOLOv8n (fine-tuned)**
- Architektura jednoetapowa: bezpośrednia predykcja bounding boxów
- Fine-tuning pre-trenowanego modelu na NEU-DET
- 40× szybszy od Faster R-CNN, niższa mAP
- Implementacja: ultralytics

**Wzorzec Strategy** — wspólny interfejs `BasePredictor`:
```python
class BasePredictor(ABC):
    def predict(self, image_path: str, threshold: float) -> dict: ...
    def predict_bytes(self, image_bytes: bytes, threshold: float) -> dict: ...
```
→ zmiana modelu bez modyfikacji API ani frontendu

---

### Slajd 7 — Wyniki i porównanie (2 min)

**Tytuł:** Wyniki ewaluacji (360 obrazów walidacyjnych)

| Model              | mAP50     | mAP50-95  | Czas/obraz |
|--------------------|-----------|-----------|------------|
| Faster R-CNN ResNet50 | **0.928** | **0.573** | ~200 ms   |
| YOLOv8n fine-tuned | 0.726     | 0.417     | ~5 ms      |

**Per-class mAP50-95:**

| Klasa            | Faster R-CNN | YOLOv8n |
|------------------|--------------|---------|
| crazing          | 0.441        | 0.152   |
| inclusion        | 0.575        | 0.435   |
| patches          | 0.686        | 0.597   |
| pitted_surface   | 0.590        | 0.542   |
| rolled-in_scale  | 0.575        | 0.229   |
| scratches        | 0.572        | 0.545   |

**Wnioski:**
- Faster R-CNN lepszy dla dokładności (mAP50: +27 pp)
- YOLOv8 lepszy dla edge/real-time (~200 ms vs ~5 ms)
- Wybór modelu zależy od wymagań aplikacji

---

### Slajd 8 — Narzędzie inżynierskie (2 min)

**Tytuł:** Integracja w narzędziu inżynierskim

**Stack:**
- Backend: FastAPI (REST API, lazy-loading modeli)
- Frontend: HTML/CSS/JS, Canvas API (bounding boxy z kolorami per-klasa)
- Deployment: `uvicorn app.main:app`

**Endpointy API:**
```
POST /predict/upload       — wgraj własne zdjęcie
POST /predict/example/{filename} — przykład z galerii
GET  /examples             — lista przykładów
```

**Parametry użytkownika:**
- Wybór modelu (Faster R-CNN / YOLOv8 pre-trained / YOLOv8 fine-tuned)
- Próg pewności (threshold, 0.0–1.0)

**Demo na żywo** — pokazać UI z przykładowymi obrazami, zmiana modelu i threshold

---

### Slajd 9 — Fine-tuning i dane publiczne (1.5 min)

**Tytuł:** Rozszerzanie zbiorów uczących — fine-tuning

**Problem generalizacji:**
- Model nauczony na NEU-DET może słabo działać na nowych typach defektów
- Zbieranie danych produkcyjnych jest kosztowne i czasochłonne

**Podejście:**
1. **Dane publiczne** (Kaggle, Roboflow, Papers with Code) jako punkt startowy
2. **Transfer learning** — pre-trenowany backbone (COCO/ImageNet)
3. **Fine-tuning** — dostrój na małym zbiorze danych z nowej domeny
4. **Augmentacja danych** — obroty, przycięcia, zmiany jasności

**Przykład z projektu:**
- YOLOv8n pre-trenowany na COCO → fine-tuning na NEU-DET (Kaggle T4 GPU)
- Wynik: mAP50 0.726 przy minimalnym koszcie treningu

**Wniosek:** fine-tuning pozwala adaptować modele do nowych zastosowań bez budowania zbioru od zera

---

### Slajd 10 — Podsumowanie (1 min)

**Tytuł:** Podsumowanie

**Zaprezentowane:**
- ✅ Pełny pipeline: dane → trening → ewaluacja → narzędzie
- ✅ Dwa modele deep learning: Faster R-CNN (dokładność) vs YOLOv8 (szybkość)
- ✅ Zbiór NEU-DET: 1800 obrazów, 6 klas defektów
- ✅ Narzędzie inżynierskie z REST API i interaktywnym UI
- ✅ Strategia fine-tuningu na danych publicznych

**Dalsze kierunki:**
- Segmentacja instancyjna (Mask R-CNN, YOLOv8-seg)
- Aktywne uczenie (active learning) z danymi produkcyjnymi
- Deployment na edge (ONNX, TensorRT)

---

## Tasks

### Phase 1: Przygotowanie treści

- [ ] Task 1.1 — Zebrać screenshoty UI (demo/model_and_examples.PNG, wyniki detekcji)
  - Depends on: nothing
- [ ] Task 1.2 — Przygotować wykresy mAP (per-class porównanie obu modeli)
  - Depends on: nothing
- [ ] Task 1.3 — Przygotować schemat architektury pipeline (diagram przepływu)
  - Depends on: nothing

### Phase 2: Tworzenie slajdów

- [ ] Task 2.1 — Zbudować slajdy w PowerPoint / Google Slides
  - Depends on: Phase 1
- [ ] Task 2.2 — Dodać demo na żywo (slajd 8) lub nagranie screencast
  - Depends on: Task 2.1

### Phase 3: Weryfikacja

- [ ] Task 3.1 — Próbna prezentacja z pomiarem czasu (cel: ≤15 min)
  - Depends on: Phase 2
- [ ] Task 3.2 — Sprawdzić, czy demo działa na maszynie prezentacyjnej
  - Depends on: Task 3.1

## Architecture Decisions

| Decision | Options Considered | Chosen | Rationale |
|---|---|---|---|
| Kolejność modeli | Faster R-CNN pierwszy / YOLO pierwszy | Faster R-CNN | Wyższa mAP — mocniejszy punkt wejścia |
| Demo | Live demo / screencast | Do decyzji | Live demo lepsze, ale ryzykowne — warto mieć backup |

## Open Questions

- [ ] Czy prezentacja w języku polskim czy angielskim?
- [ ] Czy potrzebne jest nagranie/screencast jako backup do demo?
- [ ] Jaka platforma slajdów (PowerPoint, Google Slides, Reveal.js)?

## Progress Log

| Date       | Update                                               |
|------------|------------------------------------------------------|
| 2026-05-11 | Stworzono plan prezentacji na podstawie analizy kodu |