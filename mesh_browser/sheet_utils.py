import numpy as np
from openpyxl.chart import BarChart, Reference
from openpyxl import Workbook

def _make_bar_chart(title, x_label, gap_width):
    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.gapWidth = gap_width
    chart.legend = None
    chart.x_axis.title = x_label
    chart.y_axis.title = "Number of shapes"
    chart.x_axis.delete = False  # newer openpyxl versions otherwise hide the axes
    chart.y_axis.delete = False
    chart.width, chart.height = 18, 8
    return chart


def _add_chart_from_table(ws, chart, col, n_rows, anchor):
    """Chart the table in columns col (labels) and col+1 (counts), header in row 1."""
    data = Reference(ws, min_col=col + 1, min_row=1, max_row=n_rows + 1)
    cats = Reference(ws, min_col=col, min_row=2, max_row=n_rows + 1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws.add_chart(chart, anchor)


def _write_histogram(ws, values, col, title, x_label, anchor, bins):
    """Bin `values`, write the table at `col`/`col+1`, and draw a histogram next to it."""
    counts, edges = np.histogram(values, bins=bins)

    ws.cell(row=1, column=col, value=f"{x_label} range")
    ws.cell(row=1, column=col + 1, value="Shapes")
    for r, (c, lo, hi) in enumerate(zip(counts, edges[:-1], edges[1:]), start=2):
        ws.cell(row=r, column=col, value=f"{lo:.0f}-{hi:.0f}")
        ws.cell(row=r, column=col + 1, value=int(c))

    chart = _make_bar_chart(title, x_label, gap_width=0)  # touching bars = histogram
    _add_chart_from_table(ws, chart, col, len(counts), anchor)


def _write_class_chart(ws, class_counts, col, anchor):
    """One bar per class, showing how many shapes it has."""
    ws.cell(row=1, column=col, value="Class")
    ws.cell(row=1, column=col + 1, value="Shapes")
    for r, (name, c) in enumerate(sorted(class_counts.items()), start=2):
        ws.cell(row=r, column=col, value=name)
        ws.cell(row=r, column=col + 1, value=c)

    chart = _make_bar_chart("Shapes per class", "Class", gap_width=50)
    chart.width = 24  # classes need more room for their labels
    _add_chart_from_table(ws, chart, col, len(class_counts), anchor)



def iqr_stats(values):
    """Summary statistics plus the 1.5*IQR outlier fences. Returns a list of (label, value)."""
    v = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [
        ("Mean", v.mean()),
        ("Median", median),
        ("Std dev", v.std()),
        ("Min", v.min()),
        ("Max", v.max()),
        ("Q1 (25%)", q1),
        ("Q3 (75%)", q3),
        ("IQR (Q3 - Q1)", iqr),
        ("Lower fence (Q1 - 1.5*IQR)", low),
        ("Upper fence (Q3 + 1.5*IQR)", high),
        ("Outliers below lower fence", int((v < low).sum())),
        ("Outliers above upper fence", int((v > high).sum())),
    ]

def _write_stats(ws, values, col, row, title):
    """Write the statistics table with its top-left cell at (row, col)."""
    ws.cell(row=row, column=col, value=title)
    ws.cell(row=row, column=col + 1, value="Value")
    for r, (label, value) in enumerate(iqr_stats(values), start=row + 1):
        ws.cell(row=r, column=col, value=label)
        ws.cell(row=r, column=col + 1,
                value=value if isinstance(value, int) else round(float(value), 2))

def add_summary_sheet(wb, vertex_counts, face_counts, class_counts, bins=20):
    """One extra sheet with three charts, plus statistics under the two histogram tables."""
    ws = wb.create_sheet("summary")
    _write_histogram(ws, vertex_counts, 1, "Vertex count distribution",
                     "Vertices", "J2", bins)
    _write_histogram(ws, face_counts, 4, "Face count distribution",
                     "Triangle faces", "J20", bins)
    _write_class_chart(ws, class_counts, 7, "J38")

    stats_row = bins + 3  # the histogram tables end at row bins + 1
    _write_stats(ws, vertex_counts, 1, stats_row, "Vertex statistics")
    _write_stats(ws, face_counts, 4, stats_row, "Face statistics")
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["D"].width = 28