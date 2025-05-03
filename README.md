# 🎙️ 專案介紹 | Project Description

本專案旨在實現**本機端的中文語音轉文字處理流程**，使用：
- **OpenAI Whisper 模型**：進行語音辨識。
- **PaddlePaddle + PaddleNLP 模型**：為轉錄結果加入中文標點。
- **OpenCC**：進行簡繁體中文轉換。
- 可選擇使用 CSV 字典進行術語/錯字校正，使輸出更符合實務需求。


# 🧩 功能與處理流程 | Features & Pipeline

以下為完整處理流程，並附上程式範例說明：

## 1️⃣ 音檔轉 WAV 格式

> Whisper 在處理長影音（如 MP4）時，常遺失尾段音訊，建議先轉為 `.wav` 格式 （單聲道 + 44.1kHz）。 

```python
def convert_to_wav(input_file: str):
    base = Path(input_file).stem
    output_file = Path(input_file).with_name(f"{base}.wav")
    audio = AudioSegment.from_file(input_file)
    wav_audio = audio.set_channels(1).set_frame_rate(44100)
    wav_audio.export(output_file, format="wav")
    logging.info(f"轉換 WAV 成功: {output_file}")
    return str(output_file)
```

## 2️⃣ 語音轉文字（Whisper）

> 可選模型：`tiny`、`base`、`small`、`medium`、`large`。本專案以 `large` 模型為預設，以求最佳辨識效果。
> 進行分段切割並使用 tqdm 顯示目前處理的段落與進度比例，以確認本機端轉檔狀態，避免誤認當機，也避免單次傳入過長檔案造成 GPU 不穩。

```python
def my_whisper(audio_path, segment_length=300000):
    logging.info("開始進行中文語音辨識（分段+進度提示）")
    audio = AudioSegment.from_wav(audio_path)
    segments = [audio[i:i+segment_length] for i in range(0, len(audio), segment_length)]

    # 在此加入分段數量提示
    total_segments = len(segments)
    logging.info(f"音檔已分割成 {total_segments} 個分段進行處理。")

    full_text = ""
    for idx, segment in enumerate(tqdm(segments, desc="辨識進度")):
        segment.export("temp.wav", format="wav")
        result = whisper_model.transcribe("temp.wav", language='zh')
        full_text += result["text"]

        os.remove("temp.wav")
        del segment, result
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("中文語音辨識完成")
    return full_text

```
## 3️⃣ 簡體繁體轉換機制（為 Paddle 增加辨識率）

> Paddle 的標點符號模型對簡體中文支援較好，因此需有簡體繁體的交換機制。

```python
def ch_convert(transcript, method):
    return OpenCC(method).convert(transcript)

```

## 4️⃣ 加入標點符號（Paddle NLP）

> Paddle 模型對文字長度有上限，需分段處理，再合併結果。

```python
def add_punctuation(raw_script):
    def split_text(text, max_length=200):
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]
    text_list = split_text(raw_script)
    processed_text = punc_model.add_puncs(text_list)
    logging.info('標點符號加入完成')
    return "".join(processed_text)

```

## 5️⃣ 邏輯文字校正（可選）

> 可透過外部定義的 CSV 字典進行詞彙替換，如錯字校正、專業術語微調。

```python
def fix_wording(fix_txt, csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        replace_dict = {rows[0]: rows[1] for rows in reader}
    for old_word, new_word in replace_dict.items():
        fix_txt = fix_txt.replace(old_word, new_word)
    logging.info('文字更正置換完成')
    return fix_txt.replace("-", "\n")

```

## 6️⃣ 輸出為純文字檔與 Word 檔

> 將最終文字輸出為 `.txt` 與 `.docx`，便於後續整理與分享。

```python
def convert_text(input_text, file_name, output_file_path):
    output_text = Path(output_file_path) / f"{file_name}.txt"
    with open(output_text, 'w', encoding='utf-8') as file:
        file.write(input_text)
    logging.info(f"已輸出純文字檔: {output_text}")

def convert_word(input_text, file_name, output_file_path):
    output_word = Path(output_file_path) / f"{file_name}.docx"
    doc = Document()
    doc.add_paragraph(input_text)
    doc.save(output_word)
    logging.info(f"已輸出 Word 檔: {output_word}")
```
# ⚙️ 其他進階設計與優化 | Advanced Design

## ✅ 記憶體釋放管理

每段處理完後執行：

```python
del segment, result
torch.cuda.empty_cache()
gc.collect()
```

可有效避免多段或多檔轉錄時 GPU 爆掉。
## ✅ 支援批次處理多檔

```python
file_list = ["a.mp3", "b.m4a"]
for file in file_list:
    main(file_list=[file], ...)
```

- 每筆檔案獨立處理
- 中途錯誤不影響其他檔案
- 自動刪除中繼 `.wav` 檔案

## ✅ 全程日誌輸出（logging）

每個處理步驟皆記錄於 `logging.info()`，包含：

- 檔案開始與結束時間
- 各階段狀態
- 錯誤提示（`logging.warning`）


# 🧪 環境需求 | Environment Requirements

## ✅ 建議硬體規格

| 項目 | 規格 |
|------|------|
| CPU  | Intel Core i5-13400 |
| RAM  | DDR4 32GB |
| GPU  | NVIDIA GeForce RTX 4070 |


## ✅ 建議軟體與套件版本

| 套件 | 建議版本 |
|------|----------|
| Python | 3.8 |
| CUDA | 11.8 |
| cuDNN | 8.3 |
| PyTorch | 2.0.1 (cuda118) |
| PaddlePaddle-GPU | 2.4.2 (cuda117) |
| PaddleNLP | 2.5.2 |

> ⚠️ Paddle 和 PyTorch 運行於不同 CUDA 版本，請避免同時佔用 GPU 以降低資源衝突。

---
