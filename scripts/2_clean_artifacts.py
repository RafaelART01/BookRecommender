import pandas as pd
import re


def fix_mojibake_and_punctuation(text):
    # Если значение пустое, возвращаем как есть
    if pd.isna(text):
        return text

    text = str(text)

    # 1. Исправляем кодировку (заменяем артефакты на нормальные символы)
    replacements = {
        "â€œ": '"',  # Левая двойная кавычка
        "â€": '"',  # Правая двойная кавычка
        "â€™": "'",  # Апостроф
        "â€”": "—",  # Длинное тире
        "â€“": "–",  # Короткое тире
        "â€¦": "...",  # Многоточие
        "â€˜": "'"  # Одинарная левая кавычка
    }

    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)

    # 2. Убираем "отскочившие" запятые (например: "#1 ,New York Times ,")
    # Меняем "пробел + запятая" на "запятая + пробел"
    text = re.sub(r'\s+,', ', ', text)

    # 3. Убираем множественные пробелы (если где-то образовалось 2-3 пробела подряд)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    print("1. Opening file books_final.csv...")
    # pandas прочитает файл правильно, так как он сам его создал на прошлом шаге
    df = pd.read_csv('../data/processed/books_final.csv')

    print("2. Starting cleaning of text columns from artefacts...")
    # Применяем нашу функцию к колонкам с текстом
    columns_to_clean = ['title', 'author', 'description']

    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(fix_mojibake_and_punctuation)
            print(f"   Колонка '{col}' очищена.")

    print("3. Saving results...")
    # Сохраняем с кодировкой utf-8-sig, чтобы Excel правильно понимал кавычки и запятые
    df.to_csv('../data/processed/books_final_clean.csv', index=False, encoding='utf-8-sig')

    print("Done! Open file books_final_clean.csv for results.")


if __name__ == "__main__":
    main()