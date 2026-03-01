/* ============================================================
   Image Annotation Tool – frontend application logic
   ============================================================ */

// ── State ────────────────────────────────────────────────────────────────────

let currentBatch = 0;
let totalBatches = 1;
let config = {};
let currentImageFilename = null;
let labelNames = {};
let selectedImages = new Set();
let allBatchImages = [];
let currentSortFilter = 'all';
let sortBatchesMode = false;

// ── Bootstrap ────────────────────────────────────────────────────────────────

async function loadConfig() {
    const res = await fetch('/api/config');
    config = await res.json();
    labelNames = config.labels;
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();
    document.getElementById('annotatorInput').focus();
    setupZoomSlider();
    setupBrightnessSlider();
});

// ── Annotator modal ──────────────────────────────────────────────────────────

function setAnnotator() {
    const name = document.getElementById('annotatorInput').value.trim();
    if (!name) { alert('Please enter a name'); return; }

    fetch('/api/set-annotator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotator: name }),
    }).then(async () => {
        document.getElementById('annotatorModal').classList.remove('show');
        await loadConfig();
        loadBatch(0);
    }).catch(err => {
        console.error('Error setting annotator:', err);
        alert('Error setting annotator');
    });
}

document.getElementById('annotatorInput')?.addEventListener('keypress', e => {
    if (e.key === 'Enter') setAnnotator();
});

// ── Batch loading ────────────────────────────────────────────────────────────

async function loadBatch(batchNum) {
    const offset = batchNum * config.batch_size;

    try {
        const res = await fetch(`/api/images?offset=${offset}`);
        const data = await res.json();

        allBatchImages = data.images;
        currentSortFilter = 'all';

        document.getElementById('grid').innerHTML = '';
        selectedImages.clear();
        updateSelectedCount();

        renderPlainImages(allBatchImages);

        currentBatch = batchNum;
        totalBatches = Math.ceil(data.total / config.batch_size);
        updateBatchInfo(data);
        updateStats();
        updatePaginationButtons();
    } catch (e) {
        console.error('Error loading batch:', e);
        alert('Error loading images');
    }
}

function previousBatch() {
    if (currentBatch > 0) { loadBatch(currentBatch - 1); window.scrollTo(0, 0); }
}

function nextBatch() {
    if (currentBatch < totalBatches - 1) { loadBatch(currentBatch + 1); window.scrollTo(0, 0); }
}

function updatePaginationButtons() {
    document.getElementById('prevBtn').disabled = currentBatch === 0;
    document.getElementById('nextBtn').disabled = currentBatch >= totalBatches - 1;
}

function updateBatchInfo(data) {
    document.getElementById('batchNum').textContent = currentBatch + 1;
    document.getElementById('batchTotal').textContent = totalBatches;
    document.getElementById('batchInfo').textContent =
        `Showing ${data.images.length} images (${data.offset + 1}–${data.offset + data.images.length} of ${data.total})`;
}

// ── Rendering ────────────────────────────────────────────────────────────────

function renderPlainImages(images) {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    images.forEach(img => grid.appendChild(createTile(img)));
}

/** Render images that may include group-header sentinel objects. */
function renderImages(images) {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    images.forEach(img => {
        if (img.isHeader) {
            const header = document.createElement('div');
            header.style.cssText = `
                grid-column: 1 / -1;
                padding: 1rem 0;
                margin: 1rem 0;
                border-bottom: 2px solid #333;
                font-size: 1.1rem;
                font-weight: 700;
                color: #00d4ff;
            `;
            header.textContent = `${img.label} (${img.count} images)`;
            grid.appendChild(header);
        } else {
            grid.appendChild(createTile(img));
        }
    });
}

function createTile(image) {
    const tile = document.createElement('div');
    tile.className = image.existing_label ? 'image-tile annotated' : 'image-tile';
    tile.id = `tile-${image.filename}`;
    tile.dataset.filename = image.filename;

    // Thumbnail
    const img = document.createElement('img');
    img.src = `/api/thumbnail/${encodeURIComponent(image.filename)}`;
    img.alt = image.filename;
    tile.appendChild(img);

    // Selection checkmark
    const check = document.createElement('div');
    check.className = 'selection-check';
    check.textContent = '✓';
    tile.appendChild(check);

    // Label badge (only when already annotated)
    if (image.existing_label) {
        const overlay = document.createElement('div');
        overlay.className = 'label-overlay';
        overlay.textContent =
            Object.keys(labelNames).find(k => labelNames[k] === image.existing_label) ||
            image.existing_label;
        tile.appendChild(overlay);
    }

    // Filename tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = image.filename;
    tile.appendChild(tooltip);

    tile.addEventListener('click', e => { e.stopPropagation(); toggleSelection(image.filename, tile); });
    return tile;
}

// ── Selection ────────────────────────────────────────────────────────────────

function toggleSelection(filename, tile) {
    if (selectedImages.has(filename)) {
        selectedImages.delete(filename);
        tile.classList.remove('selected');
    } else {
        selectedImages.add(filename);
        tile.classList.add('selected');
    }
    updateSelectedCount();
}

function selectAll() {
    document.querySelectorAll('.image-tile').forEach(tile => {
        selectedImages.add(tile.dataset.filename);
        tile.classList.add('selected');
    });
    updateSelectedCount();
}

function deselectAll() {
    document.querySelectorAll('.image-tile').forEach(tile => tile.classList.remove('selected'));
    selectedImages.clear();
    updateSelectedCount();
}

function updateSelectedCount() {
    document.getElementById('selectedCount').textContent = `${selectedImages.size} selected`;
}

// ── Bulk labelling ───────────────────────────────────────────────────────────

async function bulkLabel(label) {
    if (selectedImages.size === 0) { alert('Please select at least one image'); return; }

    const labelName = Object.keys(labelNames).find(k => labelNames[k] === label);
    if (!confirm(`Label ${selectedImages.size} image(s) as "${labelName}"?`)) return;

    let success = 0;
    let failed = 0;

    for (const filename of selectedImages) {
        try {
            const res = await fetch('/api/annotate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, label }),
            });

            if (res.ok) {
                const tile = document.getElementById(`tile-${filename}`);
                tile.classList.add('annotated');
                let overlay = tile.querySelector('.label-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.className = 'label-overlay';
                    tile.appendChild(overlay);
                }
                overlay.textContent = labelName || label;
                success++;
            } else {
                const error = await res.json();
                console.error(`Failed to label ${filename}:`, error.error);
                failed++;
            }
        } catch (e) {
            console.error('Error labeling:', filename, e);
            failed++;
        }
    }

    deselectAll();
    updateStats();
    alert(`✓ Labeled ${success} image(s)${failed > 0 ? `, ${failed} failed` : ''}`);
}

// ── Single-image popup menu ──────────────────────────────────────────────────

function showPopup(event) {
    const popup = document.getElementById('imagePopup');
    popup.classList.add('show');
    popup.style.position = 'fixed';
    popup.style.left = event.clientX + 'px';
    popup.style.top  = event.clientY + 'px';
}

document.addEventListener('click', e => {
    const popup = document.getElementById('imagePopup');
    if (!popup.contains(e.target) && !e.target.closest('.image-tile')) {
        popup.classList.remove('show');
    }
});

async function selectLabel(label) {
    if (!currentImageFilename) return;

    try {
        const res = await fetch('/api/annotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: currentImageFilename, label }),
        });

        if (res.ok) {
            const tile = document.getElementById(`tile-${currentImageFilename}`);
            tile.classList.add('annotated');
            let overlay = tile.querySelector('.label-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'label-overlay';
                tile.appendChild(overlay);
            }
            overlay.textContent =
                Object.keys(labelNames).find(k => labelNames[k] === label) || label;
            document.getElementById('imagePopup').classList.remove('show');
            updateStats();
        } else {
            const error = await res.json();
            alert(`Failed to save annotation: ${error.error || 'Unknown error'}`);
        }
    } catch (e) {
        console.error('Error saving annotation:', e);
        alert(`Error saving annotation: ${e.message}`);
    }
}

// ── Stats ────────────────────────────────────────────────────────────────────

async function updateStats() {
    try {
        const res  = await fetch('/api/stats');
        const data = await res.json();

        document.getElementById('totalCount').textContent     = data.total_images;
        document.getElementById('annotatedCount').textContent = data.annotated;
        document.getElementById('pendingCount').textContent   = data.pending;

        const countsHtml = Object.keys(labelNames).map(name => {
            const count = data.by_label[name] || 0;
            return `
                <div class="label-count-box label-${labelNames[name]}">
                    <div class="label-count-value">${count}</div>
                    <div class="label-count-name">${name}</div>
                </div>
            `;
        }).join('');
        document.getElementById('labelCounts').innerHTML = countsHtml;
    } catch (e) {
        console.error('Error updating stats:', e);
    }
}

function saveAnnotations() {
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            alert(
                `📊 Annotation Summary:\n\n` +
                `✓ Annotated: ${data.annotated}/${data.total_images}\n` +
                `⏳ Pending: ${data.pending}\n\n` +
                `All annotations are automatically saved to:\n` +
                `../image_labels/annotations.parquet`
            );
            const toast = document.createElement('div');
            toast.className = 'success-toast';
            toast.textContent = `✓ ${data.annotated} annotations saved!`;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        })
        .catch(err => {
            console.error('Error getting stats:', err);
            alert('Error retrieving annotation statistics');
        });
}

// ── Sort / filter ────────────────────────────────────────────────────────────

function sortImages(filter) {
    if (!allBatchImages || allBatchImages.length === 0) {
        alert('No images loaded. Please load a batch first.');
        return;
    }

    currentSortFilter = filter;
    let filtered = [...allBatchImages];

    if (filter === 'unlabeled' || filter === 'hide-labeled') {
        filtered = filtered.filter(img => !img.existing_label);
    } else if (filter !== 'all') {
        const labelNum = labelNames[filter];
        if (labelNum) {
            filtered = filtered.filter(img => img.existing_label === labelNum);
        }
    }

    renderImages(groupImagesByLabel(filtered));
    toggleSortMenu();

    if (filter === 'all') {
        document.getElementById('batchInfo').textContent = `Showing all ${filtered.length} images`;
    } else if (filter === 'unlabeled' || filter === 'hide-labeled') {
        document.getElementById('batchInfo').textContent = `Showing ${filtered.length} unlabeled images`;
    } else {
        document.getElementById('batchInfo').textContent =
            `Showing ${filtered.length} images labeled as "${filter}"`;
    }
}

function groupImagesByLabel(images) {
    const groups = { unlabeled: [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [] };

    images.forEach(img => {
        const label = img.existing_label || 'unlabeled';
        if (groups[label]) groups[label].push(img);
    });

    const ordered = [];
    ['unlabeled', '1', '2', '3', '4', '5', '6', '7'].forEach(label => {
        if (groups[label].length > 0) {
            const labelName = label === 'unlabeled'
                ? 'Unlabeled'
                : Object.keys(labelNames).find(k => labelNames[k] === label) || label;

            ordered.push({ isHeader: true, label: labelName, count: groups[label].length });
            ordered.push(...groups[label]);
        }
    });

    return ordered;
}

function toggleSortMenu() {
    const menu = document.getElementById('sortMenu');
    if (!menu) return;
    menu.classList.toggle('show');
    if (menu.classList.contains('show')) {
        menu.style.bottom = 'auto';
        menu.style.top = '100px';
    }
}

// Close sort menu when clicking outside
document.addEventListener('click', e => {
    const menu     = document.getElementById('sortMenu');
    const controls = document.querySelector('.resize-controls');
    if (menu && !menu.contains(e.target) && !controls.contains(e.target)) {
        menu.classList.remove('show');
    }
});

// ── Global batch sort mode ───────────────────────────────────────────────────

function toggleSortBatchesMode() {
    const btn = document.getElementById('sortBatchesBtn');

    fetch('/api/toggle-sort-by-label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    }).then(res => res.json())
      .then(data => {
        sortBatchesMode = data.sort_by_label_mode;
        const toast = document.createElement('div');
        toast.className = 'success-toast';

        if (sortBatchesMode) {
            btn.classList.add('active');
            btn.textContent = '✓ All Images Sorted by Label';
            toast.textContent = '✓ All images now sorted by label across batches';
        } else {
            btn.classList.remove('active');
            btn.textContent = '📊 Sort All by Label';
            toast.textContent = '◯ Sorting disabled – back to normal order';
        }

        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
        currentBatch = 0;
        loadBatch(0);
      })
      .catch(err => { console.error('Error toggling sort by label:', err); alert('Error toggling sort mode'); });
}

// ── Patient / WellFOV toggle ─────────────────────────────────────────────────

let currentLabelMode = 'patient';

function togglePatientWellFOV() {
    const btn     = document.getElementById('toggleLabelBtn');
    const infoBar = document.querySelector('.info-bar');
    const toast   = document.createElement('div');
    toast.className = 'success-toast';

    if (currentLabelMode === 'patient') {
        currentLabelMode = 'wellFOV';
        btn.textContent = '🔬 WellFOV';
        btn.classList.add('wellFOV');
        infoBar.style.borderLeftColor = '#06aed5';
        toast.textContent = '🔬 Switched to WellFOV mode';
    } else {
        currentLabelMode = 'patient';
        btn.textContent = '👤 Patient';
        btn.classList.remove('wellFOV');
        infoBar.style.borderLeftColor = '#00d4ff';
        toast.textContent = '👤 Switched to Patient mode';
    }

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2000);
}

// ── Zoom / brightness sliders ────────────────────────────────────────────────

function setupZoomSlider() {
    const slider = document.getElementById('zoomSlider');
    const label  = document.getElementById('zoomValue');
    const grid   = document.getElementById('grid');

    slider.addEventListener('input', e => {
        const size = parseInt(e.target.value);
        label.textContent = size;
        grid.style.gridTemplateColumns = `repeat(auto-fit, minmax(${size}px, 1fr))`;
        grid.style.gap = `${Math.max(0.5, Math.min(1.5, size / 100))}rem`;
    });
}

function setupBrightnessSlider() {
    const slider = document.getElementById('brightnessSlider');
    const label  = document.getElementById('brightnessValue');

    slider.addEventListener('input', e => {
        const brightness = parseInt(e.target.value);
        label.textContent = brightness;
        document.documentElement.style.setProperty('--brightness', brightness + '%');
    });
}
