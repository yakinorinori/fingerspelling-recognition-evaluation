# fingerspelling-recognition-evaluation

指文字認識の実行スクリプトと評価実験。

## 構成

### 認識実行/
- `Video_test.py` — 動画からの指文字認識（静止指文字）
- `Video_test_move.py` — 動画からの指文字認識（動作指文字）
- `Video_learn.py` — ディレクトリ内の全動画に対して Video_test.py を一括実行
- `main_jitujikan.py` — カメラ入力によるリアルタイム指文字認識

### 評価分析/
- `Evaluation_experiment.py` — 仮名ごとの検出数集計による評価実験
- `jikken_number1.py` — 実験結果 CSV のフレーム単位集計
- `code.py` — 精度・適合率・再現率・F1 スコアの比較グラフ作成

## 必要ライブラリ
`opencv-python`, `mediapipe`, `numpy`, `matplotlib`
