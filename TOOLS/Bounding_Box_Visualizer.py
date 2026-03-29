#!/usr/bin/env python3
"""
GrowLiv YOLO Dataset Viewer
────────────────────────────
Usage:
    python yolo_viewer.py                        # opens folder picker dialog
    python yolo_viewer.py /path/to/GrowLiv-Dataset

Expects:
    GrowLiv-Dataset/
        images/
            alfalfa_weevil/
                alfalfa_weevil001.jpg
        labels/
            alfalfa_weevil/
                alfalfa_weevil001.txt
        classes.txt  (optional)

Keys:
    ← / →   or  A / D    prev / next image
    ↑ / ↓   or  W / S    jump 10 images
    Enter                 send current image (+ label) to review/ folder
    L                     toggle labels on/off
    F                     fit image to window
    +/-                   zoom in/out
    Q / Esc              quit
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import shutil
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import colorsys

# ── Colour palette (per class id) ──────────────────────────────────────────
def gen_colors(n=80):
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

COLORS = gen_colors(80)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


# ── Dataset scanning ────────────────────────────────────────────────────────
def find_dataset_root(path: Path):
    """Return the directory that contains 'images/' folder."""
    if (path / 'images').is_dir():
        return path
    for child in path.iterdir():
        if child.is_dir() and (child / 'images').is_dir():
            return child
    return path  # fallback


def load_classes(root: Path):
    for name in ('classes.txt', 'obj.names', 'names.txt'):
        p = root / name
        if p.exists():
            lines = p.read_text().strip().splitlines()
            return {i: l.strip() for i, l in enumerate(lines)}
    return {}


def build_label_map(labels_dir: Path):
    """Return dict: basename -> Path to .txt file"""
    mapping = {}
    for txt in labels_dir.rglob('*.txt'):
        mapping[txt.stem] = txt
    return mapping


def scan_dataset(root: Path):
    ds_root = find_dataset_root(root)
    images_dir = ds_root / 'images'
    labels_dir = ds_root / 'labels'

    class_names = load_classes(ds_root)
    label_map = build_label_map(labels_dir) if labels_dir.is_dir() else {}

    entries = []
    for img_path in sorted(images_dir.rglob('*')):
        if img_path.suffix.lower() in IMAGE_EXTS:
            label_path = label_map.get(img_path.stem)
            folder = img_path.parent.relative_to(images_dir)
            entries.append({
                'img': img_path,
                'label': label_path,
                'folder': str(folder) if str(folder) != '.' else '',
                'name': img_path.name,
            })

    return entries, class_names, ds_root


def parse_yolo_label(txt_path: Path):
    boxes = []
    if not txt_path or not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


# ── Main App ────────────────────────────────────────────────────────────────
class YOLOViewer(tk.Tk):
    def __init__(self, dataset_path=None):
        super().__init__()
        self.title("GrowLiv YOLO Viewer")
        self.configure(bg='#0d1117')
        self.geometry('1280x800')
        self.minsize(800, 600)

        self.entries = []
        self.class_names = {}
        self.ds_root = None
        self.current_idx = 0
        self.show_labels = tk.BooleanVar(value=True)
        self.zoom = 1.0
        self.fit_mode = True
        self._photo = None
        self._orig_img = None
        self._filtered = []
        self._list_indices = []
        self._boxes = []

        self._build_ui()
        self._bind_keys()

        if dataset_path:
            self.load_dataset(Path(dataset_path))
        else:
            self.after(200, self.pick_folder)

    # ── UI Construction ──────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self, bg='#161b22', pady=6)
        topbar.pack(fill='x', side='top')

        tk.Label(topbar, text='⬛ GrowLiv YOLO Viewer', bg='#161b22',
                 fg='#3dffa0', font=('Courier', 13, 'bold')).pack(side='left', padx=12)

        self.stats_var = tk.StringVar(value='No dataset loaded')
        tk.Label(topbar, textvariable=self.stats_var, bg='#161b22',
                 fg='#5a6070', font=('Courier', 9)).pack(side='left', padx=8)

        tk.Button(topbar, text='📂 Open Folder', command=self.pick_folder,
                  bg='#3dffa0', fg='#000', font=('Courier', 9, 'bold'),
                  relief='flat', padx=8).pack(side='right', padx=10)

        # ── Main pane ──
        pane = tk.PanedWindow(self, orient='horizontal', bg='#0d1117',
                              sashwidth=4, sashrelief='flat')
        pane.pack(fill='both', expand=True)

        # ── Sidebar ──
        sidebar = tk.Frame(pane, bg='#13171c', width=240)
        pane.add(sidebar, minsize=160)

        tk.Label(sidebar, text='IMAGES', bg='#13171c', fg='#ff3966',
                 font=('Courier', 8, 'bold')).pack(anchor='w', padx=8, pady=(8,2))

        search_frame = tk.Frame(sidebar, bg='#13171c')
        search_frame.pack(fill='x', padx=6, pady=2)
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', lambda *_: self._filter_list())
        tk.Entry(search_frame, textvariable=self.search_var, bg='#0d1117',
                 fg='#e8eaed', insertbackground='#3dffa0',
                 font=('Courier', 9), relief='flat',
                 highlightthickness=1, highlightcolor='#3dffa0',
                 highlightbackground='#1e2329').pack(fill='x')

        list_frame = tk.Frame(sidebar, bg='#13171c')
        list_frame.pack(fill='both', expand=True, padx=4, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg='#1e2329', troughcolor='#0d1117',
                                 relief='flat', width=6)
        scrollbar.pack(side='right', fill='y')

        self.listbox = tk.Listbox(list_frame, bg='#0d1117', fg='#8b949e',
                                  selectbackground='#1c2d1e', selectforeground='#3dffa0',
                                  font=('Courier', 8), relief='flat',
                                  yscrollcommand=scrollbar.set,
                                  activestyle='none', borderwidth=0,
                                  highlightthickness=0)
        self.listbox.pack(fill='both', expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self._on_list_select)

        # ── Canvas area ──
        canvas_frame = tk.Frame(pane, bg='#0d1117')
        pane.add(canvas_frame, minsize=400)

        # Nav bar
        nav = tk.Frame(canvas_frame, bg='#161b22', pady=4)
        nav.pack(fill='x')

        btn_style = dict(bg='#1e2329', fg='#e8eaed', font=('Courier', 9),
                         relief='flat', padx=10, pady=2, activebackground='#3dffa0',
                         activeforeground='#000', cursor='hand2')

        btn_review_style = dict(bg='#3a1a1a', fg='#ff6b6b', font=('Courier', 9, 'bold'),
                                relief='flat', padx=10, pady=2, activebackground='#ff3966',
                                activeforeground='#fff', cursor='hand2')

        tk.Button(nav, text='◀ Prev', command=self.prev_image, **btn_style).pack(side='left', padx=4)
        tk.Button(nav, text='Next ▶', command=self.next_image, **btn_style).pack(side='left', padx=2)
        tk.Button(nav, text='−', command=self.zoom_out, **btn_style).pack(side='left', padx=2)
        tk.Button(nav, text='+', command=self.zoom_in, **btn_style).pack(side='left', padx=2)
        tk.Button(nav, text='Fit', command=self.zoom_fit, **btn_style).pack(side='left', padx=2)
        tk.Button(nav, text='→ Review  [Enter]', command=self.send_to_review,
                  **btn_review_style).pack(side='left', padx=8)

        tk.Checkbutton(nav, text='Show Labels', variable=self.show_labels,
                       command=self.redraw, bg='#161b22', fg='#8b949e',
                       selectcolor='#0d1117', activebackground='#161b22',
                       activeforeground='#3dffa0', font=('Courier', 8)).pack(side='left', padx=8)

        self.counter_var = tk.StringVar(value='')
        tk.Label(nav, textvariable=self.counter_var, bg='#161b22',
                 fg='#5a6070', font=('Courier', 8)).pack(side='right', padx=10)

        self.imgname_var = tk.StringVar(value='')
        tk.Label(nav, textvariable=self.imgname_var, bg='#161b22',
                 fg='#3dffa0', font=('Courier', 8)).pack(side='right', padx=6)

        # Canvas with scrollbars
        cv_frame = tk.Frame(canvas_frame, bg='#0d1117')
        cv_frame.pack(fill='both', expand=True)

        self.h_scroll = tk.Scrollbar(cv_frame, orient='horizontal', bg='#1e2329',
                                     troughcolor='#0d1117', relief='flat')
        self.h_scroll.pack(side='bottom', fill='x')
        self.v_scroll = tk.Scrollbar(cv_frame, bg='#1e2329',
                                     troughcolor='#0d1117', relief='flat')
        self.v_scroll.pack(side='right', fill='y')

        self.canvas = tk.Canvas(cv_frame, bg='#090d12', cursor='crosshair',
                                highlightthickness=0,
                                xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)
        self.canvas.pack(fill='both', expand=True)
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        self.canvas.bind('<Configure>', lambda e: self.redraw())

        # ── Annotation panel ──
        ann_frame = tk.Frame(pane, bg='#13171c', width=190)
        pane.add(ann_frame, minsize=140)

        tk.Label(ann_frame, text='ANNOTATIONS', bg='#13171c', fg='#3dffa0',
                 font=('Courier', 8, 'bold')).pack(anchor='w', padx=8, pady=(8,4))

        ann_scroll = tk.Scrollbar(ann_frame, bg='#1e2329', troughcolor='#0d1117',
                                  relief='flat', width=5)
        ann_scroll.pack(side='right', fill='y')

        self.ann_text = tk.Text(ann_frame, bg='#0d1117', fg='#8b949e',
                                font=('Courier', 8), relief='flat', state='disabled',
                                wrap='word', width=22, borderwidth=0,
                                highlightthickness=0,
                                yscrollcommand=ann_scroll.set)
        self.ann_text.pack(fill='both', expand=True, padx=4)
        ann_scroll.config(command=self.ann_text.yview)

        # ── Status bar ──
        self.status_var = tk.StringVar(value='Ready. Open a GrowLiv-Dataset folder.')
        tk.Label(self, textvariable=self.status_var, bg='#0d1117', fg='#5a6070',
                 font=('Courier', 8), anchor='w', pady=3).pack(fill='x', side='bottom', padx=8)

    def _bind_keys(self):
        self.bind('<Left>',   lambda e: self.prev_image())
        self.bind('<Right>',  lambda e: self.next_image())
        self.bind('<a>',      lambda e: self.prev_image())
        self.bind('<d>',      lambda e: self.next_image())
        self.bind('<Up>',     lambda e: self.jump(-10))
        self.bind('<Down>',   lambda e: self.jump(10))
        self.bind('<w>',      lambda e: self.jump(-10))
        self.bind('<s>',      lambda e: self.jump(10))
        self.bind('<Return>', lambda e: self.send_to_review())
        self.bind('<l>',      lambda e: self.show_labels.set(not self.show_labels.get()) or self.redraw())
        self.bind('<f>',      lambda e: self.zoom_fit())
        self.bind('<plus>',   lambda e: self.zoom_in())
        self.bind('<equal>',  lambda e: self.zoom_in())
        self.bind('<minus>',  lambda e: self.zoom_out())
        self.bind('<q>',      lambda e: self.quit())
        self.bind('<Escape>', lambda e: self.quit())
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind('<Button-4>', lambda e: self.zoom_in())
        self.canvas.bind('<Button-5>', lambda e: self.zoom_out())

    def _on_mousewheel(self, event):
        if event.delta > 0: self.zoom_in()
        else: self.zoom_out()

    # ── Dataset loading ──────────────────────────────────────────────────
    def pick_folder(self):
        path = filedialog.askdirectory(title='Select GrowLiv-Dataset folder')
        if path:
            self.load_dataset(Path(path))

    def load_dataset(self, path: Path):
        self.status_var.set(f'Scanning {path} ...')
        self.update()
        try:
            self.entries, self.class_names, self.ds_root = scan_dataset(path)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return

        if not self.entries:
            messagebox.showwarning('No images', 'No image files found in the images/ directory.')
            return

        self._filtered = self.entries[:]
        self.current_idx = 0
        self._populate_list()
        self._update_stats()
        self.show_image(0)
        self.status_var.set(f'Loaded {len(self.entries)} images from {path}')

    def _populate_list(self):
        self.listbox.delete(0, 'end')
        last_folder = None
        self._list_indices = []

        for i, e in enumerate(self._filtered):
            if e['folder'] != last_folder:
                last_folder = e['folder']
                label = f'📁 {e["folder"]}' if e['folder'] else '📁 (root)'
                self.listbox.insert('end', label)
                self.listbox.itemconfig('end', fg='#ff3966', selectforeground='#ff3966',
                                        selectbackground='#1e2329')
                self._list_indices.append(None)

            label = ('  ✗ ' if not e['label'] else '  ✓ ') + e['name']
            self.listbox.insert('end', label)
            color = '#5a6070' if not e['label'] else '#8b949e'
            self.listbox.itemconfig('end', fg=color)
            self._list_indices.append(i)

    def _filter_list(self):
        q = self.search_var.get().lower()
        if q:
            self._filtered = [e for e in self.entries
                              if q in e['name'].lower() or q in e['folder'].lower()]
        else:
            self._filtered = self.entries[:]
        self.current_idx = 0
        self._populate_list()
        if self._filtered:
            self.show_image(0)

    def _on_list_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = self._list_indices[sel[0]]
        if idx is None:
            return
        self.current_idx = idx
        self.show_image(idx)

    def _update_stats(self):
        total = len(self.entries)
        labeled = sum(1 for e in self.entries if e['label'])
        self.stats_var.set(f'{total} images · {labeled} labeled · {len(self.class_names)} classes')

    def _highlight_list_item(self, idx):
        for row, mapped in enumerate(self._list_indices):
            if mapped == idx:
                self.listbox.selection_clear(0, 'end')
                self.listbox.selection_set(row)
                self.listbox.see(row)
                break

    # ── Review: move current image (+ label) to review/ ─────────────────
    def send_to_review(self):
        if not self._filtered or self.ds_root is None:
            return

        e = self._filtered[self.current_idx]
        img_path: Path = e['img']
        label_path: Path = e['label']

        # Build destination paths mirroring subfolder structure
        images_dir = self.ds_root / 'images'
        labels_dir = self.ds_root / 'labels'
        review_images_dir = self.ds_root / 'review' / 'images'
        review_labels_dir = self.ds_root / 'review' / 'labels'

        try:
            rel_img = img_path.relative_to(images_dir)
        except ValueError:
            rel_img = Path(img_path.name)

        dest_img = review_images_dir / rel_img
        dest_img.parent.mkdir(parents=True, exist_ok=True)

        # Move image
        shutil.move(str(img_path), str(dest_img))
        moved_label = False

        # Move label if it exists
        if label_path and label_path.exists():
            try:
                rel_lbl = label_path.relative_to(labels_dir)
            except ValueError:
                rel_lbl = Path(label_path.name)
            dest_lbl = review_labels_dir / rel_lbl
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label_path), str(dest_lbl))
            moved_label = True

        self.status_var.set(
            f'→ review/  {img_path.name}'
            + ('  +label' if moved_label else '  (no label)')
        )

        # Remove from entries and filtered list, advance to next image
        self.entries.remove(e)
        self._filtered.remove(e)
        self._update_stats()
        self._populate_list()

        if not self._filtered:
            self._orig_img = None
            self.canvas.delete('all')
            self.counter_var.set('')
            self.imgname_var.set('')
            return

        self.current_idx = min(self.current_idx, len(self._filtered) - 1)
        self.show_image(self.current_idx)

    # ── Navigation ───────────────────────────────────────────────────────
    def prev_image(self):
        if self._filtered and self.current_idx > 0:
            self.current_idx -= 1
            self.show_image(self.current_idx)

    def next_image(self):
        if self._filtered and self.current_idx < len(self._filtered) - 1:
            self.current_idx += 1
            self.show_image(self.current_idx)

    def jump(self, delta):
        if not self._filtered:
            return
        self.current_idx = max(0, min(len(self._filtered)-1, self.current_idx + delta))
        self.show_image(self.current_idx)

    # ── Image display ────────────────────────────────────────────────────
    def show_image(self, idx):
        if not self._filtered:
            return
        e = self._filtered[idx]
        self.counter_var.set(f'{idx+1} / {len(self._filtered)}')
        name = (e['folder']+'/' if e['folder'] else '') + e['name']
        self.imgname_var.set(name)
        self._highlight_list_item(idx)

        try:
            self._orig_img = Image.open(e['img']).convert('RGB')
        except Exception as ex:
            self.status_var.set(f'Error loading image: {ex}')
            return

        self._boxes = parse_yolo_label(e['label']) if e['label'] else []
        self._update_annotation_panel(self._boxes)

        if self.fit_mode:
            self.zoom_fit(redraw=False)

        self.redraw()

        label_status = f'{len(self._boxes)} boxes' if e['label'] else 'no label file'
        self.status_var.set(f'{name}  —  {self._orig_img.width}×{self._orig_img.height}  —  {label_status}  |  Enter = send to review')

    def redraw(self):
        if self._orig_img is None:
            return

        w = int(self._orig_img.width * self.zoom)
        h = int(self._orig_img.height * self.zoom)

        img = self._orig_img.resize((w, h), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        for cls, cx, cy, bw, bh in self._boxes:
            color = COLORS[cls % len(COLORS)]
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            for t in range(2):
                draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)

            if self.show_labels.get():
                label = self.class_names.get(cls, f'cls {cls}')
                font_size = max(10, min(16, int((x2-x1) / max(len(label),1) * 1.2)))
                try:
                    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', font_size)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0,0), label, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                tx, ty = x1, max(0, y1 - th - 4)
                draw.rectangle([tx, ty, tx+tw+6, ty+th+4], fill=color)
                draw.text((tx+3, ty+2), label, fill='black', font=font)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))

    # ── Zoom ─────────────────────────────────────────────────────────────
    def zoom_in(self):
        self.fit_mode = False
        self.zoom = min(self.zoom * 1.2, 10.0)
        self.redraw()

    def zoom_out(self):
        self.fit_mode = False
        self.zoom = max(self.zoom / 1.2, 0.05)
        self.redraw()

    def zoom_fit(self, redraw=True):
        if self._orig_img is None:
            return
        self.fit_mode = True
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        iw, ih = self._orig_img.width, self._orig_img.height
        self.zoom = min(cw / iw, ch / ih, 1.0)
        if redraw:
            self.redraw()

    # ── Annotation panel ─────────────────────────────────────────────────
    def _update_annotation_panel(self, boxes):
        self.ann_text.config(state='normal')
        self.ann_text.delete('1.0', 'end')
        if not boxes:
            self.ann_text.insert('end', 'No annotations\n', 'muted')
        else:
            for i, (cls, cx, cy, bw, bh) in enumerate(boxes):
                label = self.class_names.get(cls, f'Class {cls}')
                color = COLORS[cls % len(COLORS)]
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                self.ann_text.tag_configure(f'cls_{cls}', foreground=hex_color)
                self.ann_text.insert('end', f'[{i}] {label}\n', f'cls_{cls}')
                self.ann_text.tag_configure('dim', foreground='#5a6070')
                self.ann_text.insert('end',
                    f'  cx {cx:.4f}  cy {cy:.4f}\n'
                    f'  w  {bw:.4f}  h  {bh:.4f}\n\n', 'dim')
        self.ann_text.config(state='disabled')


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    app = YOLOViewer(dataset_path=path)
    app.mainloop()