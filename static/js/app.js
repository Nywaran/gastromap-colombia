// === GastroMap Colombia — Map & Interactivity ===

let map, markersLayer;
let allData = [];

function getColor(gtpi) {
    if (gtpi >= 80) return '#1a9641';
    if (gtpi >= 60) return '#a6d96a';
    if (gtpi >= 40) return '#fee08b';
    if (gtpi >= 20) return '#fdae61';
    return '#d7191c';
}

function getRadius(count) {
    if (count >= 500) return 14;
    if (count >= 100) return 11;
    if (count >= 50) return 9;
    if (count >= 10) return 7;
    return 5;
}

function classifStyle(classification) {
    const styles = {
        'Very High': 'background:rgba(26,150,65,.2);color:#1a9641',
        'High':      'background:rgba(166,217,106,.2);color:#7cb342',
        'Medium':    'background:rgba(254,224,139,.25);color:#b8860b',
        'Low':       'background:rgba(253,174,97,.2);color:#ef6c00',
        'Very Low':  'background:rgba(215,25,28,.2);color:#d7191c'
    };
    return styles[classification] || '';
}

function createPopup(m) {
    const subcats = m.top_subcategories && m.top_subcategories.length > 0
        ? m.top_subcategories.slice(0, 3).join(', ')
        : 'N/A';
    return `
        <div style="min-width:220px">
            <div class="popup-title">${m.muni_name}</div>
            <div class="popup-dept">${m.dept_name} · ${m.region}</div>
            <div>
                <span class="popup-score">${m.GTPI}</span>
                <span class="popup-clasif" style="${classifStyle(m.classification)}">${m.classification}</span>
            </div>
            <div class="popup-details">
                <div>🍽️ <strong>${m.gastro_count}</strong> gastronomy establishments</div>
                <div>📊 <strong>${m.gastro_variety}</strong> distinct types</div>
                <div>🏨 <strong>${m.total_providers}</strong> tourism providers</div>
                <div>🌡️ ~${m.avg_temperature.toFixed(1)}°C · ${m.precipitation} mm/yr</div>
                <div style="margin-top:4px;font-size:11px;opacity:.7">Top: ${subcats}</div>
            </div>
        </div>
    `;
}

function renderMarkers(data) {
    if (markersLayer) map.removeLayer(markersLayer);
    markersLayer = L.layerGroup();
    data.forEach(m => {
        const marker = L.circleMarker([m.latitude, m.longitude], {
            radius: getRadius(m.gastro_count),
            fillColor: getColor(m.GTPI),
            color: 'rgba(255,255,255,.3)',
            weight: 1,
            fillOpacity: 0.85
        });
        marker.bindPopup(createPopup(m), { maxWidth: 300 });
        markersLayer.addLayer(marker);
    });
    markersLayer.addTo(map);
}

async function loadData(params = '') {
    try {
        const resp = await fetch(`/api/municipalities${params}`);
        const data = await resp.json();
        allData = data;
        renderMarkers(data);
    } catch (err) {
        console.error('Error loading data:', err);
    }
}

async function loadDepartments() {
    try {
        const resp = await fetch('/api/departments');
        const departments = await resp.json();
        const select = document.getElementById('filterDept');
        departments.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error('Error loading departments:', err);
    }
}

function applyFilters() {
    const region = document.getElementById('filterRegion').value;
    const dept = document.getElementById('filterDept').value;
    const classif = document.getElementById('filterClassification').value;
    const minGtpi = document.getElementById('filterGTPI').value;
    const params = new URLSearchParams();
    if (region) params.set('region', region);
    if (dept) params.set('department', dept);
    if (classif) params.set('classification', classif);
    if (minGtpi > 0) params.set('min_gtpi', minGtpi);
    const query = params.toString();
    loadData(query ? `?${query}` : '');
}

function resetFilters() {
    document.getElementById('filterRegion').value = '';
    document.getElementById('filterDept').value = '';
    document.getElementById('filterClassification').value = '';
    document.getElementById('filterGTPI').value = 0;
    document.getElementById('gtpiValue').textContent = '0';
    loadData();
}

// Range slider live display
document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('filterGTPI');
    const display = document.getElementById('gtpiValue');
    if (slider && display) {
        slider.addEventListener('input', () => {
            display.textContent = slider.value;
        });
    }
});

// Sortable table columns
let sortDir = {};
function sortTable(colIndex) {
    const table = document.getElementById('deptTable');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    sortDir[colIndex] = !sortDir[colIndex];
    const dir = sortDir[colIndex] ? 1 : -1;
    rows.sort((a, b) => {
        let va = a.cells[colIndex].textContent.trim();
        let vb = b.cells[colIndex].textContent.trim();
        const na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return (na - nb) * dir;
        return va.localeCompare(vb) * dir;
    });
    rows.forEach(r => tbody.appendChild(r));
}

// Table search
document.addEventListener('DOMContentLoaded', () => {
    const search = document.getElementById('searchDept');
    if (search) {
        search.addEventListener('input', () => {
            const query = search.value.toLowerCase();
            const rows = document.querySelectorAll('#deptTable tbody tr');
            rows.forEach(r => {
                r.style.display = r.cells[0].textContent.toLowerCase().includes(query) ? '' : 'none';
            });
        });
    }
});

// Initialize Leaflet map
document.addEventListener('DOMContentLoaded', () => {
    const mapEl = document.getElementById('map');
    if (!mapEl) return; // Only runs on dashboard page

    map = L.map('map', { zoomControl: true, scrollWheelZoom: true }).setView([4.6, -74.1], 6);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> · Data: DANE, MinCIT, IDEAM',
        maxZoom: 18
    }).addTo(map);
    loadData();
    loadDepartments();
});
