"""
CİK-SÖR V1 Demo — Akustik Erken Uyarı Sistemi
================================================
Tavuk seslerini analiz ederek normal vokalizasyon, anormal/hastalık
şüphesi taşıyan vokalizasyon ve çevresel gürültüyü ayıran çalışan
bir yapay zekâ prototipi.

Çalıştırma: python app.py
"""

import os
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Proje kökünü ayarla
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.utils import setup_logging, ensure_directories, timestamp
from src.audio_io import load_audio, record_from_mic
from src.preprocessing import preprocess, pad_or_trim
from src.noise_filter import reduce_noise
from src.segmentation import segment_audio
from src.features import extract_rf_features
from src.spectrogram import create_spectrogram, create_spectrogram_image
from src.predictor import ModelManager
from src.decision import evaluate_prediction, confidence_text
from src.database import DatabaseManager

# ═══════════════════════════════════════════════════════════
#  BAŞLATMA
# ═══════════════════════════════════════════════════════════
setup_logging()
logger = logging.getLogger("ciksor.app")

logger.info("CİK-SÖR V1 Demo başlatılıyor...")
ensure_directories()

models = ModelManager()
db = DatabaseManager()

logger.info("Sistem hazır.")


# ═══════════════════════════════════════════════════════════
#  ANA ANALİZ PIPELINE
# ═══════════════════════════════════════════════════════════
def analyze_audio(y, sr, source_type="dosya", file_name=None):
    """
    Tam analiz pipeline:
    1. Ön işleme → 2. Gürültü azaltma → 3. Segmentleme →
    4. Özellik çıkarma → 5. Tahmin → 6. Karar → 7. Loglama

    Returns: dict (tüm sonuçlar)
    """
    if y is None or len(y) == 0:
        return {"error": "Ses verisi boş."}

    # 1. Ön işleme
    y_raw = y.copy()
    y = preprocess(y, sr)
    y = pad_or_trim(y, sr)

    # 2. Gürültü azaltma
    y_clean, noise_stats = reduce_noise(y, sr)

    # 3. Segmentleme (tek segment için doğrudan devam)
    segments = segment_audio(y_clean, sr)
    if not segments:
        return {"error": "Ses çok sessiz veya anlamsız, analiz yapılamadı."}

    seg = segments[0]  # İlk segmenti analiz et

    # 4+5. Tahmin
    result_rf = None
    result_cnn = None
    result_tflite = None
    spec_array = None

    # RF tahmin
    try:
        features = extract_rf_features(seg, sr)
        result_rf = models.predict_rf(features)
    except Exception as e:
        logger.error(f"RF pipeline hatası: {e}")

    # CNN tahmin (spektrogram gerekli)
    try:
        spec_array = create_spectrogram(seg, sr)
        result_cnn = models.predict_cnn(spec_array)
    except Exception as e:
        logger.error(f"CNN pipeline hatası: {e}")

    # TFLite tahmin
    if spec_array is not None:
        try:
            result_tflite = models.predict_tflite(spec_array)
        except Exception as e:
            logger.error(f"TFLite pipeline hatası: {e}")

    # Mock fallback
    if result_rf is None and result_cnn is None and models.mock_mode:
        mock_result = models.predict_mock(seg)
        result_cnn = mock_result  # Mock'u CNN yerine kullan

    # 6. Karar (CNN öncelikli, yoksa RF)
    if result_cnn:
        primary_class, primary_conf = result_cnn
    elif result_rf:
        primary_class, primary_conf = result_rf
    else:
        primary_class, primary_conf = "Noise", 0.0

    decision = evaluate_prediction(primary_class, primary_conf)

    # 7. Loglama
    db.log_prediction(
        predicted_class=primary_class,
        confidence=primary_conf,
        source_type=source_type,
        file_name=file_name,
        alert=decision["alert"],
        notes=decision.get("alert_message"),
    )

    return {
        "decision": decision,
        "rf": result_rf,
        "cnn": result_cnn,
        "tflite": result_tflite,
        "noise_stats": noise_stats,
        "raw_audio": y_raw,
        "clean_audio": y_clean,
        "spectrogram": spec_array,
        "mock_mode": models.mock_mode,
    }


# ═══════════════════════════════════════════════════════════
#  GÖRSELLEŞTİRME YARDIMCILARI
# ═══════════════════════════════════════════════════════════
def plot_waveform(y, sr, title="Ses Dalgası", color="#2ecc71"):
    """Ses dalgası grafiği oluşturur."""
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, color=color, linewidth=0.5, alpha=0.9)
    ax.fill_between(t, y, alpha=0.15, color=color)
    ax.set_xlim(0, len(y) / sr)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(title, color="white", fontsize=10)
    ax.set_xlabel("Süre (s)", color="#aaa", fontsize=8)
    ax.tick_params(colors="#666", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.grid(True, alpha=0.1, color="#555")
    plt.tight_layout()
    return fig


def plot_spectrogram_fig(spec_array):
    """Spektrogram görsel figür oluşturur."""
    if spec_array is None:
        return None
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.imshow(spec_array.astype(np.uint8), aspect="auto")
    ax.set_title("3-Kanallı Spektrogram (RGB)", color="white", fontsize=10)
    ax.tick_params(colors="#666", labelsize=7)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  GRADIO ARAYÜZ HANDLERLERİ
# ═══════════════════════════════════════════════════════════
def handle_file_analysis(file_path):
    """Dosya ile test sekmesi handler'ı."""
    if not file_path:
        return "⚠ Lütfen bir ses veya video dosyası yükleyin.", None, None, None, ""

    y, sr, err = load_audio(file_path)
    if err:
        return f"❌ Hata: {err}", None, None, None, ""

    result = analyze_audio(y, sr, source_type="dosya",
                           file_name=os.path.basename(file_path))

    if "error" in result:
        return f"❌ {result['error']}", None, None, None, ""

    d = result["decision"]
    mock_note = "\n⚠ Demo/Mock Mode aktif — gerçek model tahmini yapılmıyor." if result["mock_mode"] else ""

    # Sonuç metni
    conf_txt = confidence_text(d["confidence_level"])
    result_text = (
        f"### {d['display_label']}\n\n"
        f"**Güven Skoru:** %{d['confidence']*100:.1f} ({conf_txt})\n\n"
    )
    if result["rf"]:
        result_text += f"**RF Motoru:** {result['rf'][0]} (%{result['rf'][1]*100:.1f})\n\n"
    if result["cnn"]:
        result_text += f"**CNN Motoru:** {result['cnn'][0]} (%{result['cnn'][1]*100:.1f})\n\n"
    if result["tflite"]:
        result_text += f"**Edge AI:** {result['tflite'][0]} (%{result['tflite'][1]*100:.1f}, {result['tflite'][2]:.1f}ms)\n\n"

    ns = result["noise_stats"]
    result_text += f"**Gürültü Azaltma:** %{ns['reduction_pct']:.1f}\n"
    result_text += mock_note

    # Alert banner
    alert_html = ""
    if d["alert"]:
        alert_html = (
            f'<div style="background:#2d0a0e;border:1px solid #ff475740;'
            f'border-left:4px solid #e74c3c;border-radius:8px;padding:14px 20px;'
            f'margin:8px 0;color:#ff8a80;font-weight:600;">'
            f'{d["alert_message"]}</div>'
        )

    # Görseller
    wave_raw = plot_waveform(result["raw_audio"], sr, "Ham Ses", "#f39c12")
    wave_clean = plot_waveform(result["clean_audio"], sr, "Temizlenmiş Ses",
                               d["color"])
    spec_fig = plot_spectrogram_fig(result["spectrogram"])

    return result_text, alert_html, wave_raw, wave_clean, spec_fig


def handle_mic_analysis():
    """Canlı mikrofon analizi handler'ı."""
    y, sr, err = record_from_mic()
    if err:
        return f"❌ Hata: {err}", None, None, None, ""

    result = analyze_audio(y, sr, source_type="mikrofon", file_name="Canlı Kayıt")

    if "error" in result:
        return f"❌ {result['error']}", None, None, None, ""

    d = result["decision"]
    mock_note = "\n⚠ Demo/Mock Mode aktif — gerçek model tahmini yapılmıyor." if result["mock_mode"] else ""

    conf_txt = confidence_text(d["confidence_level"])
    result_text = (
        f"### {d['display_label']}\n\n"
        f"**Güven Skoru:** %{d['confidence']*100:.1f} ({conf_txt})\n"
        + mock_note
    )

    alert_html = ""
    if d["alert"]:
        alert_html = (
            f'<div style="background:#2d0a0e;border:1px solid #ff475740;'
            f'border-left:4px solid #e74c3c;border-radius:8px;padding:14px 20px;'
            f'margin:8px 0;color:#ff8a80;font-weight:600;">'
            f'{d["alert_message"]}</div>'
        )

    wave_raw = plot_waveform(result["raw_audio"], sr, "Ham Ses", "#f39c12")
    wave_clean = plot_waveform(result["clean_audio"], sr, "Temizlenmiş Ses", d["color"])
    spec_fig = plot_spectrogram_fig(result["spectrogram"])

    return result_text, alert_html, wave_raw, wave_clean, spec_fig


def handle_logs_refresh():
    """Log tablosu yenileme handler'ı."""
    logs = db.get_recent_logs(limit=30)
    if not logs:
        return "Henüz analiz kaydı yok."

    rows = []
    for l in logs:
        display = config.DISPLAY_LABELS.get(l["predicted_class"], l["predicted_class"])
        alert_icon = "🔴" if l["alert"] else ""
        rows.append(
            f"| {l['id']} | {l['timestamp']} | {l['source_type'] or '-'} | "
            f"{l['file_name'] or '-'} | {display} | "
            f"%{l['confidence']*100:.1f} | {alert_icon} |"
        )

    header = "| # | Zaman | Kaynak | Dosya | Tahmin | Güven | Uyarı |\n|---|---|---|---|---|---|---|\n"
    return header + "\n".join(rows)


def handle_alert_logs():
    """Sadece uyarı logları."""
    logs = db.get_alert_logs(limit=20)
    if not logs:
        return "Uyarı kaydı bulunmuyor."

    rows = []
    for l in logs:
        rows.append(
            f"| {l['id']} | {l['timestamp']} | {l['file_name'] or '-'} | "
            f"%{l['confidence']*100:.1f} | {l['notes'] or '-'} |"
        )

    header = "| # | Zaman | Dosya | Güven | Not |\n|---|---|---|---|---|\n"
    return header + "\n".join(rows)


def handle_system_status():
    """Sistem durumu bilgisi."""
    status = models.get_status()
    log_counts = db.get_log_count()

    mic_ok = True
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        mic_ok = any(d["max_input_channels"] > 0 for d in devices)
    except Exception:
        mic_ok = False

    lines = [
        f"**RF Modeli:** {'✅ Yüklü' if status['rf'] else '❌ Bulunamadı'}",
        f"**CNN Modeli:** {'✅ Yüklü' if status['cnn'] else '❌ Bulunamadı'}",
        f"**TFLite Modeli:** {'✅ Yüklü' if status['tflite'] else '⬜ Yok (opsiyonel)'}",
        f"**Demo/Mock Mode:** {'🟡 AKTİF — gerçek model tahmini yapılmıyor' if status['mock_mode'] else '🟢 Pasif (gerçek model aktif)'}",
        "",
        f"**Mikrofon:** {'✅ Erişilebilir' if mic_ok else '❌ Erişilemedi'}",
        f"**Veritabanı:** ✅ Çalışıyor ({config.DB_PATH})",
        f"**Toplam Analiz:** {log_counts['total']}",
        f"**Uyarı Sayısı:** {log_counts['alerts']}",
        f"**Son Kontrol:** {timestamp()}",
        "",
        f"**Örnekleme Hızı:** {config.SR} Hz",
        f"**Segment Süresi:** {config.SEGMENT_DURATION}s",
    ]
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  GRADIO ARAYÜZ
# ═══════════════════════════════════════════════════════════
def build_ui():
    """5 sekmeli Gradio dashboard oluşturur."""
    import gradio as gr

    header_html = """
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 28px;
    border-radius:12px;border:1px solid #2a2a4a;margin-bottom:12px;">
        <h1 style="margin:0;color:#e8e8e8;font-size:24px;font-weight:700;">
            🐔 CİK-SÖR — Akustik Erken Uyarı Sistemi
        </h1>
        <p style="margin:4px 0 0;color:#8888aa;font-size:13px;">
            Tavuk vokalizasyonlarını analiz ederek anormal sesleri tespit eden yapay zekâ prototipi
        </p>
    </div>
    """

    disclaimer = (
        "> **Not:** Bu sonuç kesin teşhis değildir; erken uyarı amaçlıdır. "
        "Nihai model, saha donanımıyla toplanacak veteriner onaylı verilerle "
        "yeniden eğitilecektir."
    )

    with gr.Blocks(
        title="CİK-SÖR Akustik Erken Uyarı",
    ) as demo:

        gr.HTML(header_html)

        with gr.Tabs():

            # ──────── SEKME 1: CANLI ANALİZ ────────
            with gr.Tab("🎙️ Canlı Analiz"):
                gr.Markdown("Mikrofondan **5 saniyelik** kayıt alarak analiz yapın.")
                mic_btn = gr.Button("🎙️ Kayıt Al ve Analiz Et", variant="primary", size="lg")
                mic_result = gr.Markdown()
                mic_alert = gr.HTML()
                with gr.Row():
                    mic_wave_raw = gr.Plot(label="Ham Ses")
                    mic_wave_clean = gr.Plot(label="Temizlenmiş Ses")
                mic_spec = gr.Plot(label="Spektrogram")
                gr.Markdown(disclaimer)

                mic_btn.click(
                    fn=handle_mic_analysis,
                    outputs=[mic_result, mic_alert, mic_wave_raw, mic_wave_clean, mic_spec],
                )

            # ──────── SEKME 2: DOSYA İLE TEST ────────
            with gr.Tab("📁 Dosya ile Test"):
                gr.Markdown("WAV, MP3 veya MP4 dosyası yükleyerek analiz yapın.")
                file_input = gr.Audio(type="filepath", label="Ses/Video Dosyası Yükle")
                file_btn = gr.Button("🔍 Analiz Et", variant="primary")
                file_result = gr.Markdown()
                file_alert = gr.HTML()
                with gr.Row():
                    file_wave_raw = gr.Plot(label="Ham Ses")
                    file_wave_clean = gr.Plot(label="Temizlenmiş Ses")
                file_spec = gr.Plot(label="Spektrogram")
                gr.Markdown(disclaimer)

                file_btn.click(
                    fn=handle_file_analysis,
                    inputs=[file_input],
                    outputs=[file_result, file_alert, file_wave_raw,
                             file_wave_clean, file_spec],
                )

            # ──────── SEKME 3: SİNYAL GÖRSELLEŞTİRME ────────
            with gr.Tab("📊 Sinyal Görselleştirme"):
                gr.Markdown(
                    "Bir ses dosyası yükleyin; ham dalga, temizlenmiş dalga ve "
                    "spektrogram görüntülenecektir."
                )
                viz_input = gr.Audio(type="filepath", label="Ses Dosyası")
                viz_btn = gr.Button("📊 Görselleştir", variant="primary")
                viz_result = gr.Markdown()
                viz_alert = gr.HTML()
                with gr.Row():
                    viz_wave_raw = gr.Plot(label="Ham Ses Dalgası")
                    viz_wave_clean = gr.Plot(label="Temizlenmiş Ses Dalgası")
                viz_spec = gr.Plot(label="3-Kanallı Spektrogram")

                viz_btn.click(
                    fn=handle_file_analysis,
                    inputs=[viz_input],
                    outputs=[viz_result, viz_alert, viz_wave_raw,
                             viz_wave_clean, viz_spec],
                )

            # ──────── SEKME 4: LOGLAR ────────
            with gr.Tab("📋 Loglar"):
                gr.Markdown("### Analiz Geçmişi")
                log_btn = gr.Button("🔄 Yenile", variant="secondary")
                log_table = gr.Markdown("Henüz analiz kaydı yok.")

                gr.Markdown("### Uyarı Logları")
                alert_log_btn = gr.Button("🔴 Uyarıları Göster", variant="secondary")
                alert_table = gr.Markdown()

                log_btn.click(fn=handle_logs_refresh, outputs=[log_table])
                alert_log_btn.click(fn=handle_alert_logs, outputs=[alert_table])

            # ──────── SEKME 5: SİSTEM DURUMU ────────
            with gr.Tab("⚙️ Sistem Durumu"):
                gr.Markdown("### Sistem Kontrolü")
                status_btn = gr.Button("🔄 Durumu Kontrol Et", variant="secondary")
                status_out = gr.Markdown()

                status_btn.click(fn=handle_system_status, outputs=[status_out])

                # Açılışta otomatik kontrol
                demo.load(fn=handle_system_status, outputs=[status_out])

    return demo


# ═══════════════════════════════════════════════════════════
#  BAŞLAT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 55)
    print("  CİK-SÖR V1 Demo — Akustik Erken Uyarı Sistemi")
    print("=" * 55)

    app = build_ui()

    import gradio as gr

    print("[OK] Arayüz hazır, sunucu başlatılıyor...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
    )