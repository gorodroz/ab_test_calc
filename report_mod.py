from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import os

styles = getSampleStyleSheet()


def generate_report(results_dict, graphs_paths, filename="ab_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []

    story.append(Paragraph("A/B Testing Report", styles["Title"]))
    story.append(Spacer(1, 20))

    for section, text in results_dict.items():
        story.append(Paragraph(f"<b>{section}</b>", styles["Heading2"]))
        story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Graphs</b>", styles["Heading2"]))
    for path in graphs_paths:
        if os.path.exists(path):
            story.append(Image(path, width=400, height=250))
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"\n>>> PDF report generated: {filename}")
