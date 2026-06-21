import os
import json
import zipfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def export_raw_csv(df: pd.DataFrame, file_path: str) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path

def export_preprocessed_csv(
    time_array: np.ndarray,
    preprocessed_signal: np.ndarray,
    vpg: np.ndarray,
    apg: np.ndarray,
    file_path: str
) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df = pd.DataFrame({
        "time_sec": time_array,
        "preprocessed_signal": preprocessed_signal,
        "vpg_1st_derivative": vpg,
        "apg_2nd_derivative": apg
    })
    df.to_csv(file_path, index=False)
    return file_path

def export_features_csv(features: Dict[str, Any], file_path: str) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert dict to a 1-row DataFrame
    df = pd.DataFrame([features])
    df.to_csv(file_path, index=False)
    return file_path

def export_report_json(report_dict: Dict[str, Any], file_path: str) -> str:
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Handle numpy arrays in serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, cls=NumpyEncoder, indent=4)
    return file_path

def create_research_package(
    export_files: List[str], 
    zip_path: str,
    manifest_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Creates a ZIP packaging containing all generated CSV/JSON reports,
    plus an automatically compiled manifest.json.
    """
    zip_path = os.path.abspath(zip_path)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    # Create manifest
    manifest = {
        "package_type": "BioPulse Research Package",
        "verdict_wording": "Ready for biomedical research",
        "files_included": [os.path.basename(f) for f in export_files]
    }
    if manifest_info:
        manifest.update(manifest_info)
        
    temp_manifest_path = os.path.join(os.path.dirname(zip_path), "manifest.json")
    with open(temp_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    all_files = export_files + [temp_manifest_path]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in all_files:
            if os.path.exists(file):
                zip_file.write(file, os.path.basename(file))
                
    # Clean up temp manifest file
    if os.path.exists(temp_manifest_path):
        os.remove(temp_manifest_path)
        
    return zip_path

def export_png_plot(
    time_array: np.ndarray,
    raw_signal: np.ndarray,
    preprocessed_signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    file_path: str
) -> str:
    import matplotlib.pyplot as plt
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # We only plot the first 10 seconds (or min of 10s and signal duration) by default to match screen view
    max_idx = min(len(time_array), int(10 * fs))
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_array[:max_idx], raw_signal[:max_idx], color='#163B6D', label='Raw Signal')
    plt.title('Raw PPG Waveform (First 10 Seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_array[:max_idx], preprocessed_signal[:max_idx], color='#16A34A', label='Preprocessed')
    if len(peaks) > 0:
        # Filter peaks to first 10 seconds
        peaks_in_range = peaks[peaks < max_idx]
        if len(peaks_in_range) > 0:
            plt.scatter(time_array[peaks_in_range], preprocessed_signal[peaks_in_range], color='#EF4444', marker='^', s=40, label='Systolic Peaks', zorder=3)
    plt.title('Preprocessed PPG Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()
    return file_path

def export_pdf_report(res: Dict[str, Any], png_path: str, file_path: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#0F2747'),
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#163B6D'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#1F2937'),
        leading=12
    )
    bold_body_style = ParagraphStyle(
        'BoldBodyStyle',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("BioPulse Studio X — Research Report", title_style))
    story.append(Paragraph("Universal Photoplethysmography Signal Analytics Platform", body_style))
    story.append(Spacer(1, 10))
    
    # KPI Table
    kpi_data = [
        [
            Paragraph("<b>File Name:</b>", body_style), Paragraph(str(res['file_info']['file_name']), body_style),
            Paragraph("<b>Sampling Rate:</b>", body_style), Paragraph(f"{res['fs']:.1f} Hz", body_style)
        ],
        [
            Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{res['duration_sec']:.1f} s", body_style),
            Paragraph("<b>Total Samples:</b>", body_style), Paragraph(str(len(res['raw_sig'])), body_style)
        ],
        [
            Paragraph("<b>Signal Quality:</b>", body_style), Paragraph(f"{res['quality_score']:.1f}% ({res['quality_band']})", body_style),
            Paragraph("<b>Readiness Score:</b>", body_style), Paragraph(f"{res['readiness_score']:.1f}% (Cat {res['category']})", body_style)
        ],
        [
            Paragraph("<b>E2E Verdict:</b>", body_style), Paragraph(f"<b>{res['verdict'].overall_verdict}</b>", bold_body_style),
            Paragraph("<b>Wording:</b>", body_style), Paragraph(str(res['verdict'].wording), body_style)
        ]
    ]
    
    t_kpi = Table(kpi_data, colWidths=[90, 170, 90, 170])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F6F8FB')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DCE3EC')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    
    story.append(t_kpi)
    story.append(Spacer(1, 10))
    
    # Embedded Waveform Chart
    if png_path and os.path.exists(png_path):
        story.append(Paragraph("Biosignal Waveform Comparison (First 10s)", heading_style))
        story.append(Image(png_path, width=480, height=240))
        story.append(Spacer(1, 10))
        
    # Quality Metrics Table
    story.append(Paragraph("Signal Quality Sub-Metrics Details", heading_style))
    qm = res["quality_metrics"]
    qm_data = [
        [Paragraph("<b>Metric Name</b>", bold_body_style), Paragraph("<b>Value</b>", bold_body_style)],
        [Paragraph("Signal-to-Noise Ratio (SNR)", body_style), Paragraph(f"{qm['snr_db']:.2f} dB", body_style)],
        [Paragraph("Baseline Wander Ratio", body_style), Paragraph(f"{qm['drift_ratio']:.4f}", body_style)],
        [Paragraph("Motion Artifact Spikes %", body_style), Paragraph(f"{qm['motion_artifact_pct']:.2f}%", body_style)],
        [Paragraph("Signal Entropy", body_style), Paragraph(f"{qm['entropy']:.4f}", body_style)],
        [Paragraph("Flatline Regions %", body_style), Paragraph(f"{qm['flat_region_pct']:.2f}%", body_style)],
        [Paragraph("Missing Data / NaNs %", body_style), Paragraph(f"{qm['missing_data_pct']:.2f}%", body_style)],
        [Paragraph("Peak Amplitude Stability", body_style), Paragraph(f"{qm['peak_stability']:.4f}", body_style)],
        [Paragraph("Beat Consistency", body_style), Paragraph(f"{qm['beat_consistency']:.4f}", body_style)],
        [Paragraph("Total Peaks Count", body_style), Paragraph(str(qm['peak_count']), body_style)]
    ]
    
    t_qm = Table(qm_data, colWidths=[240, 240])
    t_qm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#E8EEF5')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DCE3EC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_qm)
    
    doc.build(story)
    return file_path
