import csv

# CSVファイルを開く
with open('jikken.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    frame = None
    for i, row in enumerate(reader):
        if i % 6 == 0:  # 6行ごとに新しいフレームが開始する
            frame = row[0]
        elif i % 6 == 1:  # フレームの次の行には1位の文字とその確率がある
            first_rank_line = row[0] + row[1] + row[2]
            print(f'{frame}\n{first_rank_line}')
