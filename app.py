import os
import torch
import shutil
import librosa
import warnings
import numpy as np
import gradio as gr
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
from model import EvalNet
from utils import (
    get_modelist,
    find_wav_files,
    embed_img,
    _L,
    SAMPLE_RATE,
    TEMP_DIR,
    TRANSLATE,
    CLASSES,
)


def wav2mel(audio_path: str, width=1.6, topdb=40):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    non_silents = librosa.effects.split(y, top_db=topdb)
    non_silent = np.concatenate([y[start:end] for start, end in non_silents])
    mel_spec = librosa.feature.melspectrogram(y=non_silent, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    dur = librosa.get_duration(y=non_silent, sr=sr)
    total_frames = log_mel_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_mel_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{TEMP_DIR}/mel_{round(dur, 2)}_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def wav2cqt(audio_path: str, width=1.6, topdb=40):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    non_silents = librosa.effects.split(y, top_db=topdb)
    non_silent = np.concatenate([y[start:end] for start, end in non_silents])
    cqt_spec = librosa.cqt(y=non_silent, sr=sr)
    log_cqt_spec = librosa.power_to_db(np.abs(cqt_spec) ** 2, ref=np.max)
    dur = librosa.get_duration(y=non_silent, sr=sr)
    total_frames = log_cqt_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_cqt_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{TEMP_DIR}/cqt_{round(dur, 2)}_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def wav2chroma(audio_path: str, width=1.6, topdb=40):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    non_silents = librosa.effects.split(y, top_db=topdb)
    non_silent = np.concatenate([y[start:end] for start, end in non_silents])
    chroma_spec = librosa.feature.chroma_stft(y=non_silent, sr=sr)
    log_chroma_spec = librosa.power_to_db(np.abs(chroma_spec) ** 2, ref=np.max)
    dur = librosa.get_duration(y=non_silent, sr=sr)
    total_frames = log_chroma_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_chroma_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{TEMP_DIR}/chroma_{round(dur, 2)}_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def most_common_element(input_list: list):
    counter = Counter(input_list)
    mce, _ = counter.most_common(1)[0]
    return mce


def infer(wav_path: str, log_name: str, folder_path=TEMP_DIR):
    status = "Success"
    filename = result = None
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        if not wav_path:
            raise ValueError("请输入音频!")

        spec = log_name.split("_")[-3]
        os.makedirs(folder_path, exist_ok=True)
        model = EvalNet(log_name, len(TRANSLATE)).model
        eval("wav2%s" % spec)(wav_path)
        outputs = []
        all_files = os.listdir(folder_path)
        for file_name in all_files:
            if file_name.lower().endswith(".jpg"):
                file_path = os.path.join(folder_path, file_name)
                input = embed_img(file_path)
                output: torch.Tensor = model(input)
                pred_id = torch.max(output.data, 1)[1]
                outputs.append(int(pred_id))

        max_count_item = most_common_element(outputs)
        shutil.rmtree(folder_path)
        filename = os.path.basename(wav_path)
        result = TRANSLATE[CLASSES[max_count_item]]

    except Exception as e:
        status = f"{e}"

    return status, filename, result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    models = get_modelist(assign_model="GoogleNet_mel")
    examples = []
    example_wavs = find_wav_files()
    for wav in example_wavs:
        examples.append([wav, models[0]])

    with gr.Blocks() as demo:
        gr.Interface(
            fn=infer,
            inputs=[
                gr.Audio(label=_L("上传录音 (>40dB)"), type="filepath"),
                gr.Dropdown(choices=models, label=_L("选择模型"), value=models[0]),
            ],
            outputs=[
                gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                gr.Textbox(label=_L("音频文件名"), buttons=["copy"]),
                gr.Textbox(label=_L("唱法识别"), buttons=["copy"]),
            ],
            examples=examples,
            cache_examples=False,
            flagging_mode="never",
            title=_L("建议录音时长保持在 5s 左右, 过长会影响识别效率"),
        )

        gr.Markdown(
            f"# {_L('引用')}"
            + """
            ```bibtex
            @article{Zhou-2025,
                author  = {Monan Zhou and Shenyang Xu and Zhaorui Liu and Zhaowen Wang and Feng Yu and Wei Li and Baoqiang Han},
                title   = {CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research},
                journal = {Transactions of the International Society for Music Information Retrieval},
                volume  = {8},
                number  = {1},
                pages   = {22--38},
                month   = {Mar},
                year    = {2025},
                url     = {https://doi.org/10.5334/tismir.194},
                doi     = {10.5334/tismir.194}
            }
            ```"""
        )

    demo.launch(css="#gradio-share-link-button-0 { display: none; }", ssr_mode=False)
