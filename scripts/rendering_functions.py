import matplotlib.pyplot as plt
import six
import numpy as np


def highlight_max(s):
    is_max = s == s.max()
    # bold = 'bold' if val < 0 else ''
    return ['font-weight: bold' if v else '' for v in is_max]


def highlight_min(s):
    is_min = s == s.min()
    # bold = 'bold' if val < 0 else ''
    return ['font-weight: bold' if v else '' for v in is_min]


def render_mpl_table(data, title, path_to_save, col_width=3.0,
                     row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                     edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])
                ) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox,
                         colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    ax.set_title(title)
    plt.savefig(path_to_save)

    return ax
