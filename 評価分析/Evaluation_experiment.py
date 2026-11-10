import os
import csv
import re


def count_files_with_kana(directory, target_kana, kana_list, min_number=0, max_number=579):
    count = 0
    other_kana_counts = {}
    total_files = 0

    for filename in os.listdir(directory):
        total_files += 1
        number_part = re.findall(r'\d+', filename)
        if number_part:
            number = int(number_part[0])
            if min_number <= number <= max_number:
                found_kana = None
                for kana in kana_list:
                    if kana in filename:
                        found_kana = kana
                        break
                if found_kana == target_kana:
                    count += 1
                else:
                    other_kana_counts[found_kana] = other_kana_counts.get(found_kana, 0) + 1

    return count, other_kana_counts, total_files


def process_csv_ranges(csv_file, directory, kana_list):
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            target_kana = row[0]
            min_number = int(row[1])
            max_number = int(row[2])

            count, other_kana_counts, total_files = count_files_with_kana(directory, target_kana, kana_list, min_number,
                                                                          max_number)

            print(f"指定されたカナ文字 '{target_kana}' のファイルの個数 ({min_number}-{max_number}): {count}")
            print("指定された範囲内に含まれる他のカナ文字とその数:")
            for kana, count in other_kana_counts.items():
                print(f"  カナ文字 '{kana}' のファイルの個数: {count}")
            print(f"ディレクトリ内の全画像の数: {total_files}\n")


# 五十音リスト
kana_list = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を", "ん"
]

# # 使用例
# csv_file = "path/to/your/range.csv"
# directory = "path/to/your/files/directory"
# process_csv_ranges(csv_file, directory, kana_list)

# 使用例
csv_file = "./評価実験/実験04_再実験/実験04_1.csv"
directory = "./exfinger"
process_csv_ranges(csv_file, directory, kana_list)
