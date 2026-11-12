# Лабораторная работа 2. Сверточные сети: Pet Vision Lab

Вся работа лежит в одном ноутбуке `Lab_2_CNN_Pet_Vision.ipynb`. Данные: Oxford-IIIT Pet (7 349 фотографий, 37 пород кошек и собак, маски сегментации).

## Что в ноутбуке

1. Свёртка, нулевое дополнение и пулинг на чистом NumPy, проверка против PyTorch, фильтры Собеля.
2. Базовая сеть `SimpleCNN` с нуля.
3. ResNet-18 с весами ImageNet: feature extraction и fine-tuning.
4. U-Net для сегментации (питомец / фон / граница), IoU и Dice, ablation без skip-связей.
5. Grad-CAM и анализ ошибок.
6. Бонус: зависимость качества от размера обучающей выборки.

## Что лежит в папке

| Файл | Зачем |
|---|---|
| `Lab_2_CNN_Pet_Vision.ipynb` | ноутбук с кодом, результатами и выводами |
| `data/oxford-iiit-pet/images.tar.gz`, `annotations.tar.gz` | архивы датасета (в репозитории, ~810 МБ) |
| `.gitignore` | распакованные папки `images/` и `annotations/` не коммитятся |
| `.gitattributes` | архивы хранятся через Git LFS (файл больше 100 МБ) |

## Как запустить

### 1. Окружение

Нужен Python 3.10 или новее. Я запускал на Python 3.13 с такими версиями: torch 2.14, torchvision 0.29, numpy 2.3, pandas 3.0, matplotlib 3.11, Pillow 11.3.

```bash
pip install torch torchvision numpy pandas matplotlib pillow jupyter
```

Для видеокарты NVIDIA ставьте сборку PyTorch с CUDA по инструкции на pytorch.org. Без GPU всё тоже работает, но обучение занимает намного больше времени.

### 2. Данные

Если репозиторий клонирован вместе с LFS, архивы уже лежат в `data/oxford-iiit-pet/`:

```bash
git lfs install
git clone <адрес репозитория>
```

Распаковывать вручную ничего не нужно: при первом запуске первой ячейки `torchvision` проверит архивы и сам распакует их в `images/` и `annotations/`. Если архивов нет, он скачает датасет (около 800 МБ), при занятом сервере Oxford ячейку нужно перезапустить.

### 3. Запуск ноутбука

```bash
jupyter notebook Lab_2_CNN_Pet_Vision.ipynb
```

В меню: Kernel → Restart & Run All. Либо без интерфейса:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Lab_2_CNN_Pet_Vision.ipynb
```

На Windows перед этим удобно выставить `PYTHONIOENCODING=utf8`.

## Что важно знать

- **Время.** Полный прогон занял у меня около 35 минут на одной видеокарте NVIDIA. Больше всего времени уходит на чтение и аугментацию фотографий процессором.
- **Воспроизводимость.** Зерно `SEED = 42`, режим cuDNN детерминированный, поэтому при повторном запуске числа совпадают (я проверил двумя прогонами).
- **Windows.** В ноутбуке `NUM_WORKERS = 0` на Windows (на Linux и Colab 2). Иначе `DataLoader` не может передать в процессы классы, объявленные в ноутбуке.
- **Тестовая выборка.** Официальный test не используется ни для обучения, ни для подбора гиперпараметров, ни для ранней остановки. Все решения принимаются по валидации (20 % от trainval).
- **Colab.** В ноутбуке ничего не привязано к моей машине. В Colab нужно выбрать GPU (Runtime → Change runtime type → T4) и запустить все ячейки подряд.

## Итоговые результаты

| Модель | Val acc, % | Test acc, % |
|---|---|---|
| Baseline CNN (часть 2) | 19,2 | 16,6 |
| ResNet-18, feature extraction (3.1) | 83,8 | 83,5 |
| ResNet-18, fine-tuning (3.2) | 86,8 | 85,4 |

U-Net на тесте: pet IoU 0,778, background IoU 0,874, mean Dice 0,808. Без skip-связей: 0,767, 0,861, 0,794.
