# Домашнее задание: VLM для визуально-математического рассуждения

В этом проекте нужно реализовать упрощённый пайплайн VLM: изображение с графиком, схемой, таблицей или геометрической фигурой + текстовый вопрос -> текстовый ответ.

Задание специально разделено на три уровня данных:

| Набор | Где лежит | Зачем нужен | Обязательно? |
|---|---|---|---|
| **toy_math_vqa** | `assets/toy_math_vqa/` | Быстрая проверка, что код, форматы, processor/model/train/benchmark работают | Да |
| **math_vqa_medium** | `assets/math_vqa_medium/` | Более содержательная синтетическая практика и отчёт без внешних скачиваний | Рекомендуется |
| **MathVista testmini** | скачивается отдельно с Hugging Face | Проверка качества / benchmark на профильном visual math наборе | Для расширенного трека / бонуса |

Главное: **toy-набор не является датасетом для оценки качества**. Он нужен как smoke-check, чтобы public tests быстро запускались на CPU. Для содержательной оценки качества используйте MathVista testmini.

## Треки по железу

| Трек | Ресурс у студента | Что обязательно | Что не обязательно |
|---|---:|---|---|
| **A. CPU-only** | GPU нет | Реализовать код, пройти unit/smoke tests на toy-наборе | Обучать VLM до качества |
| **B. Small GPU** | 6–12 GB VRAM | Adapter-only обучение на маленьком/medium math subset | LoRA и большой benchmark |
| **C. A100-20GB** | 1/4 A100, около 20 GB VRAM | Adapter pretrain + SFT с LoRA, MathVista eval | Rank 256 и тяжёлый leaderboard |

Основная оценка ставится за корректный инженерный пайплайн. Качество на MathVista/hidden math benchmark используется только для расширенного трека или бонуса.


## Оценивание

Задание оценивается в **10 баллов**. Критерии: `dataset.py` — 1.5, `processor.py` — 2.0, `model.py` — 2.0, `train.py` — 1.0, `benchmark.py` — 1.0, public tests — 1.0, отчёт — 1.0. Подробности см. в [`GRADING.md`](GRADING.md).

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q tests_public
```

Для CPU-трека достаточно добиться прохождения public-тестов и написать короткий отчёт.

## Что нужно реализовать

Файлы с `TODO` находятся в папке `hw/`:

```text
hw/dataset.py      # загрузка math-VQA примеров
hw/processor.py    # preprocessing изображений, prompt, labels, collate
hw/model.py        # adapter, visual-token merge, forward/generate
hw/train.py        # training loop, сохранение adapter/checkpoint
hw/benchmark.py    # prompt для benchmark, parse ответа, accuracy
```

Запрещено менять интерфейсы функций и классов, которые используются в `tests_public/`.

## Данные

### 1. Toy-набор: только проверка работоспособности

```text
assets/toy_math_vqa/
  manifest.jsonl
  images/*.png
```

Этот набор используется public tests. Он маленький специально: тесты должны быстро запускаться на CPU и в GitHub Actions.

### 2. Medium-набор: локальная практика

```text
assets/math_vqa_medium/
  manifest.jsonl
  images/*.png
```

Это синтетический набор побольше. Его можно использовать для отчёта, проверки train loop и первых экспериментов без скачивания внешних данных.

### 3. MathVista: проверка качества

MathVista **не включён в репозиторий**. Его нужно скачать отдельно, если вы делаете расширенный трек или quality evaluation:

```bash
python -m pip install -e ".[ml]"
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 200
python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml
```

Не коммитьте скачанный MathVista в GitHub. Папка `assets/mathvista_testmini/` добавлена в `.gitignore`.

## Команды по трекам

### Track A: CPU-only

```bash
pytest -q tests_public
python -m hw.train --config configs/track_a_cpu.yaml --fast-train
python -m hw.benchmark --config configs/inference_math.yaml --toy
```

### Track B: Small GPU

```bash
python -m hw.train --config configs/track_b_small_gpu.yaml
python -m hw.benchmark --config configs/inference_math.yaml
```

### Track C: A100-20GB / quality evaluation

```bash
python -m hw.train --config configs/track_c_a100_pretrain.yaml
python -m hw.train --config configs/track_c_a100_sft.yaml
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 1000
python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml
```

## Что сдавать

План минимум:

```text
hw/*.py
report_template.md
```

Для GPU-треков дополнительно:

```text
artifacts/adapter.pt или artifacts/adapter.safetensors
artifacts/special_tokens.pt, если вы обучали новые visual special tokens
artifacts/lora или artifacts/model.pt, если вы делали SFT с LoRA
```

Не добавляйте в репозиторий большие файлы без разрешения преподавателя. Для больших чекпойнтов используйте место сдачи, указанное в LMS.
