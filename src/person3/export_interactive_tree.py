"""
Generate an interactive HTML decision tree viewer using D3.js.
Person 3 - Extra
"""
import sys, io, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]
MODEL_FILE = BASE_DIR / "model" / "baseline_model.joblib"
TRAIN_FILE = BASE_DIR / "dataset" / "online_shoppers_train.csv"
OUT_GINI   = BASE_DIR / "tree_interactive.html"

# ── Load ────────────────────────────────────────────────────
model    = joblib.load(MODEL_FILE)
train_df = pd.read_csv(TRAIN_FILE)
TARGET        = "Revenue"
feature_names = [c for c in train_df.columns if c != TARGET]
class_names   = ["No Purchase", "Purchase"]

# ── Convert sklearn tree to JSON dict ──────────────────────
tree_ = model.tree_

def build_node(node_id, depth=0):
    feat    = tree_.feature[node_id]
    thresh  = tree_.threshold[node_id]
    samples = int(tree_.n_node_samples[node_id])
    values  = tree_.value[node_id][0].tolist()
    impurity = float(tree_.impurity[node_id])
    majority = int(np.argmax(values))

    node = {
        "id"       : int(node_id),
        "depth"    : depth,
        "samples"  : samples,
        "values"   : [int(v) for v in values],
        "impurity" : round(impurity, 4),
        "class"    : class_names[majority],
        "class_id" : majority,
    }

    if feat != -2:   # not a leaf
        fname = feature_names[feat]
        # Strip preprocessing prefix for readability
        short = fname.replace("remainder__", "").replace("categorical__", "")
        node["feature"]   = fname
        node["short"]     = short
        node["threshold"] = round(float(thresh), 4)
        node["name"]      = f"{short} <= {thresh:.4f}"
        node["children"]  = [
            build_node(tree_.children_left[node_id],  depth+1),
            build_node(tree_.children_right[node_id], depth+1),
        ]
    else:
        node["name"]     = class_names[majority]
        node["is_leaf"]  = True

    return node

print("Building tree JSON...")
tree_json = build_node(0)
tree_data = json.dumps(tree_json)
print(f"Total nodes: {tree_.node_count}")

# ── HTML template ───────────────────────────────────────────
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Decision Tree — Interactive Viewer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Topbar ── */
  #topbar {
    background: #1a1d2e;
    border-bottom: 1px solid #2a2d3e;
    padding: 10px 18px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  #topbar h1 {
    font-size: 15px;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: .5px;
    margin-right: 8px;
  }
  .ctrl-group { display: flex; align-items: center; gap: 8px; }
  .ctrl-label { font-size: 12px; color: #888; }

  .btn {
    padding: 5px 13px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: .15s;
  }
  .btn-purple { background: #7c3aed; color: #fff; }
  .btn-purple:hover { background: #6d28d9; }
  .btn-teal   { background: #0d9488; color: #fff; }
  .btn-teal:hover { background: #0f766e; }
  .btn-gray   { background: #374151; color: #d1d5db; }
  .btn-gray:hover { background: #4b5563; }

  input[type=range] { accent-color: #7c3aed; cursor: pointer; width: 110px; }

  #depth-val {
    min-width: 22px;
    text-align: center;
    font-size: 13px;
    font-weight: 700;
    color: #a78bfa;
  }

  /* ── Legend ── */
  #legend {
    display: flex; gap: 14px; margin-left: auto; align-items: center;
  }
  .leg { display: flex; align-items: center; gap: 5px; font-size: 12px; }
  .leg-dot { width: 12px; height: 12px; border-radius: 3px; }

  /* ── SVG canvas ── */
  #canvas { flex: 1; overflow: hidden; position: relative; }
  svg { width: 100%; height: 100%; }

  /* ── Nodes ── */
  .node circle {
    stroke-width: 1.5px;
    cursor: pointer;
    transition: r .15s, filter .15s;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,.5));
  }
  .node circle:hover { filter: brightness(1.3) drop-shadow(0 3px 8px rgba(0,0,0,.8)); }
  .node text {
    font-size: 10px;
    fill: #e2e8f0;
    pointer-events: none;
    text-anchor: middle;
    dominant-baseline: central;
  }
  .node.leaf circle { stroke-dasharray: 3,2; }

  /* ── Links ── */
  .link {
    fill: none;
    stroke: #334155;
    stroke-width: 1.5px;
    opacity: .7;
  }

  /* ── Edge labels ── */
  .edge-label {
    font-size: 9px;
    fill: #64748b;
    pointer-events: none;
  }

  /* ── Tooltip ── */
  #tooltip {
    position: absolute;
    background: #1e2235;
    border: 1px solid #3b4260;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity .15s;
    max-width: 280px;
    z-index: 100;
    box-shadow: 0 8px 32px rgba(0,0,0,.6);
  }
  #tooltip h3 { font-size: 13px; margin-bottom: 6px; color: #a78bfa; }
  #tooltip table { border-collapse: collapse; width: 100%; }
  #tooltip td { padding: 2px 6px; }
  #tooltip td:first-child { color: #94a3b8; }
  #tooltip td:last-child  { color: #f1f5f9; font-weight: 600; text-align: right; }

  /* ── Stats bar ── */
  #stats {
    position: absolute;
    bottom: 12px; left: 50%;
    transform: translateX(-50%);
    background: rgba(26,29,46,.9);
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 6px 18px;
    font-size: 11px;
    color: #64748b;
    pointer-events: none;
    white-space: nowrap;
  }
  #stats b { color: #a78bfa; }
</style>
</head>
<body>

<div id="topbar">
  <h1>🌳 Decision Tree Viewer</h1>

  <div class="ctrl-group">
    <span class="ctrl-label">Max depth:</span>
    <input type="range" id="depth-slider" min="1" max="26" value="4"/>
    <span id="depth-val">4</span>
  </div>

  <button class="btn btn-purple" id="btn-expand">Expand All</button>
  <button class="btn btn-gray"   id="btn-collapse">Collapse</button>
  <button class="btn btn-teal"   id="btn-reset">Reset View</button>

  <div id="legend">
    <div class="leg"><div class="leg-dot" style="background:#f97316"></div>No Purchase</div>
    <div class="leg"><div class="leg-dot" style="background:#3b82f6"></div>Purchase</div>
    <div class="leg"><div class="leg-dot" style="background:#64748b; border:1px dashed #94a3b8"></div>Collapsed</div>
  </div>
</div>

<div id="canvas">
  <svg id="svg"></svg>
  <div id="tooltip"></div>
  <div id="stats">Click a node to expand/collapse &nbsp;|&nbsp; Scroll to zoom &nbsp;|&nbsp; Drag to pan</div>
</div>

<script>
const RAW = TREE_DATA_PLACEHOLDER;

// ── Config ──────────────────────────────────────────────────
const NODE_R    = 14;
const DX        = 220;   // horizontal gap
const DY        = 58;    // vertical gap per level

// ── Color ───────────────────────────────────────────────────
const classColor = d => d.data.class_id === 0 ? "#f97316" : "#3b82f6";
const strokeColor= d => d.data.class_id === 0 ? "#fb923c" : "#60a5fa";

// ── SVG / zoom setup ────────────────────────────────────────
const svg = d3.select("#svg");
const g   = svg.append("g");

const zoom = d3.zoom()
  .scaleExtent([.02, 3])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// ── Tree layout ─────────────────────────────────────────────
const layout = d3.tree().nodeSize([DX, DY]);

// ── State: build root once, track collapsed ──────────────────
function clone(obj) { return JSON.parse(JSON.stringify(obj)); }

let root;
function buildRoot(maxDepth) {
  function prune(node, d) {
    if (d >= maxDepth && node.children) {
      node._children = node.children;
      delete node.children;
    } else if (node.children) {
      node.children.forEach(c => prune(c, d+1));
    }
  }
  const data = clone(RAW);
  prune(data, 0);
  root = d3.hierarchy(data, d => d.children);
  root.x0 = 0; root.y0 = 0;
}

// ── Draw ─────────────────────────────────────────────────────
const linkGen = d3.linkVertical()
  .x(d => d.x)
  .y(d => d.y);

function update(source) {
  layout(root);

  const nodes = root.descendants();
  const links = root.links();

  // ── Links ──
  const link = g.selectAll(".link").data(links, d => d.target.data.id);

  link.enter().append("path")
    .attr("class","link")
    .attr("d", () => {
      const o = {x: source.x0, y: source.y0};
      return linkGen({source: o, target: o});
    })
    .merge(link)
    .transition().duration(350)
    .attr("d", linkGen);

  link.exit().transition().duration(350)
    .attr("d", () => {
      const o = {x: source.x, y: source.y};
      return linkGen({source: o, target: o});
    }).remove();

  // ── Edge labels (True / False) ──
  const eLbl = g.selectAll(".edge-label").data(links, d => d.target.data.id);

  eLbl.enter().append("text")
    .attr("class","edge-label")
    .attr("x", d => (d.source.x + d.target.x)/2 + (d.target === d.source.children?.[0] ? -12 : 12))
    .attr("y", d => (d.source.y + d.target.y)/2)
    .text(d => d.target === d.source.children?.[0] ? "T" : "F")
    .merge(eLbl)
    .transition().duration(350)
    .attr("x", d => (d.source.x + d.target.x)/2 + (d.target === d.source.children?.[0] ? -12 : 12))
    .attr("y", d => (d.source.y + d.target.y)/2);

  eLbl.exit().remove();

  // ── Nodes ──
  const node = g.selectAll(".node").data(nodes, d => d.data.id);

  const nodeEnter = node.enter().append("g")
    .attr("class", d => "node" + (d.data.is_leaf ? " leaf" : ""))
    .attr("transform", () => `translate(${source.x0},${source.y0})`)
    .on("click", (event, d) => toggle(d))
    .on("mouseover", showTip)
    .on("mousemove", moveTip)
    .on("mouseout",  hideTip);

  nodeEnter.append("circle")
    .attr("r", 0)
    .attr("fill", classColor)
    .attr("stroke", strokeColor);

  nodeEnter.append("text")
    .attr("dy", "0.31em")
    .text(d => d.data.is_leaf ? "🍂" :
               (d.data._children || (!d.data.children && !d.data.is_leaf)) ? "+" : "");

  const nodeUpdate = nodeEnter.merge(node);

  nodeUpdate.transition().duration(350)
    .attr("transform", d => `translate(${d.x},${d.y})`);

  nodeUpdate.select("circle")
    .transition().duration(350)
    .attr("r", d => d.data.is_leaf ? NODE_R * .8 : NODE_R)
    .attr("fill", d => {
      if (!d.data.children && !d.data.is_leaf) return "#374151";   // collapsed
      return classColor(d);
    })
    .attr("stroke", d => {
      if (!d.data.children && !d.data.is_leaf) return "#6b7280";
      return strokeColor(d);
    });

  nodeUpdate.select("text")
    .text(d => d.data.is_leaf ? "🍂" :
               (!d.data.children) ? "▶" : "");

  node.exit().transition().duration(350)
    .attr("transform", () => `translate(${source.x},${source.y})`)
    .remove();

  nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });

  // ── Auto-center on first draw ──
  if (source === root) centerView();
}

function centerView() {
  const nodes = root.descendants();
  const xs = nodes.map(d=>d.x), ys = nodes.map(d=>d.y);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const W = +svg.node().clientWidth;
  const H = +svg.node().clientHeight;
  const treeW = maxX-minX+NODE_R*4, treeH = maxY-minY+NODE_R*4;
  const scale = Math.min(W/treeW, H/treeH, 1.5) * .85;
  const tx = W/2 - scale*(minX+maxX)/2;
  const ty = 40 - scale*minY;
  svg.call(zoom.transform, d3.zoomIdentity.translate(tx,ty).scale(scale));
}

function toggle(d) {
  if (d.data.is_leaf) return;
  if (d.data.children) {
    d.data._children = d.data.children;
    delete d.data.children;
    // sync hierarchy
    d.children = null;
  } else if (d.data._children) {
    d.data.children = d.data._children;
    delete d.data._children;
    d.children = d.data.children.map(c => {
      const h = d3.hierarchy(c, x => x.children);
      h.parent = d; h.depth = d.depth+1;
      h.x0 = d.x; h.y0 = d.y;
      return h;
    });
  }
  // Rebuild hierarchy properly
  root = d3.hierarchy(RAW_snapshot(), x => x.children);
  mirrorCollapsed(root, collapseMap());
  update(d);
}

// simpler toggle: rebuild from current data state
function collapseMap() {
  const map = {};
  function walk(n) {
    if (!n.data.children && n.data._children) map[n.data.id] = true;
    (n.children||[]).forEach(walk);
  }
  walk(root);
  return map;
}

// ── Simpler approach: store collapse state by id ─────────────
let collapsedIds = new Set();

function toggle2(d) {
  if (d.data.is_leaf) return;
  const id = d.data.id;
  if (collapsedIds.has(id)) collapsedIds.delete(id);
  else collapsedIds.add(id);
  rebuild();
  update(d);
}

function rebuild() {
  function applyCollapse(node) {
    if (!node.children) return;
    if (collapsedIds.has(node.id)) {
      node._children = node.children;
      delete node.children;
    } else {
      node.children.forEach(applyCollapse);
    }
  }
  const data = clone(RAW);
  applyCollapse(data);
  const prevRoot = root;
  root = d3.hierarchy(data, d => d.children);
  root.x0 = prevRoot.x0; root.y0 = prevRoot.y0;
}

// Replace toggle in event with toggle2
g.on("click.toggle", null);   // reset

// ── Re-init with toggle2 ─────────────────────────────────────
function init(maxDepth) {
  collapsedIds.clear();
  // Collect nodes deeper than maxDepth and collapse them
  function collectDeep(node, d) {
    if (d >= maxDepth) { collapsedIds.add(node.id); return; }
    if (node.children) node.children.forEach(c => collectDeep(c, d+1));
  }
  collectDeep(RAW, 0);

  rebuild();
  g.selectAll("*").remove();
  update(root);
  centerView();
}

// ── Tooltip ──────────────────────────────────────────────────
const tip = document.getElementById("tooltip");

function showTip(event, d) {
  const dd = d.data;
  const total = dd.values[0] + dd.values[1];
  const pct0 = total ? (dd.values[0]/total*100).toFixed(1) : 0;
  const pct1 = total ? (dd.values[1]/total*100).toFixed(1) : 0;

  let html = `<h3>${dd.is_leaf ? "🍂 Leaf" : "🔀 Split"}: ${dd.name}</h3><table>`;
  html += `<tr><td>Node ID</td><td>#${dd.id}</td></tr>`;
  html += `<tr><td>Depth</td><td>${dd.depth}</td></tr>`;
  html += `<tr><td>Samples</td><td>${dd.samples.toLocaleString()}</td></tr>`;
  html += `<tr><td>No Purchase</td><td>${dd.values[0].toLocaleString()} (${pct0}%)</td></tr>`;
  html += `<tr><td>Purchase</td><td>${dd.values[1].toLocaleString()} (${pct1}%)</td></tr>`;
  html += `<tr><td>Gini</td><td>${dd.impurity}</td></tr>`;
  html += `<tr><td>Predict</td><td style="color:${dd.class_id===0?'#fb923c':'#60a5fa'}">${dd.class}</td></tr>`;
  if (!dd.is_leaf && dd.feature) {
    html += `<tr><td>Feature</td><td style="font-size:10px">${dd.short}</td></tr>`;
    html += `<tr><td>Threshold</td><td>${dd.threshold}</td></tr>`;
  }
  html += "</table>";
  tip.innerHTML = html;
  tip.style.opacity = 1;
  moveTip(event);
}
function moveTip(event) {
  const canvas = document.getElementById("canvas");
  const rect = canvas.getBoundingClientRect();
  let x = event.clientX - rect.left + 14;
  let y = event.clientY - rect.top  - 10;
  if (x + 290 > rect.width)  x -= 310;
  if (y + 200 > rect.height) y -= 180;
  tip.style.left = x + "px";
  tip.style.top  = y + "px";
}
function hideTip() { tip.style.opacity = 0; }

// ── Controls ─────────────────────────────────────────────────
const slider = document.getElementById("depth-slider");
const depthVal= document.getElementById("depth-val");
slider.addEventListener("input", () => {
  depthVal.textContent = slider.value;
  init(+slider.value);
});

document.getElementById("btn-expand").addEventListener("click", () => {
  collapsedIds.clear();
  rebuild();
  g.selectAll("*").remove();
  update(root);
  centerView();
});
document.getElementById("btn-collapse").addEventListener("click", () => {
  init(1);
  slider.value = 1; depthVal.textContent = "1";
});
document.getElementById("btn-reset").addEventListener("click", centerView);

// ── Attach toggle2 via event delegation ──────────────────────
g.on("click", (event, d) => {
  if (d) toggle2(d);
});

// ── Handle node click via selectAll ──────────────────────────
// (d3 bubbles, so we attach per-node in update)
function update(source) {
  layout(root);
  const nodes = root.descendants();
  const links = root.links();

  // Links
  const link = g.selectAll(".link").data(links, d => d.target.data.id);
  link.enter().append("path").attr("class","link")
    .attr("d", () => { const o={x:source.x0,y:source.y0}; return linkGen({source:o,target:o}); })
    .merge(link).transition().duration(300).attr("d", linkGen);
  link.exit().transition().duration(300)
    .attr("d", () => { const o={x:source.x,y:source.y}; return linkGen({source:o,target:o}); }).remove();

  // Edge labels
  const eLbl = g.selectAll(".edge-label").data(links, d => d.target.data.id);
  eLbl.enter().append("text").attr("class","edge-label")
    .merge(eLbl).transition().duration(300)
    .attr("x", d => (d.source.x+d.target.x)/2 + (d.target===d.source.children?.[0]?-14:14))
    .attr("y", d => (d.source.y+d.target.y)/2)
    .text(d => d.target===d.source.children?.[0] ? "True" : "False");
  eLbl.exit().remove();

  // Nodes
  const node = g.selectAll(".node").data(nodes, d => d.data.id);

  const ne = node.enter().append("g")
    .attr("class", d => "node"+(d.data.is_leaf?" leaf":""))
    .attr("transform", () => `translate(${source.x0},${source.y0})`)
    .on("click", (event, d) => { event.stopPropagation(); toggle2(d); })
    .on("mouseover", showTip).on("mousemove", moveTip).on("mouseout", hideTip);

  ne.append("circle").attr("r",0).attr("fill", classColor).attr("stroke", strokeColor);
  ne.append("text").attr("dy","0.31em");

  const nu = ne.merge(node);
  nu.transition().duration(300).attr("transform", d=>`translate(${d.x},${d.y})`);
  nu.select("circle").transition().duration(300)
    .attr("r", d => d.data.is_leaf ? NODE_R*.75 : NODE_R)
    .attr("fill", d => (!d.children && !d.data.is_leaf) ? "#374151" : classColor(d))
    .attr("stroke", d => (!d.children && !d.data.is_leaf) ? "#6b7280" : strokeColor(d));
  nu.select("text").text(d => d.data.is_leaf ? "🍂" : (!d.children ? "▶" : ""));

  node.exit().transition().duration(300)
    .attr("transform", () => `translate(${source.x},${source.y})`).remove();

  nodes.forEach(d => { d.x0=d.x; d.y0=d.y; });
}

// ── Start ────────────────────────────────────────────────────
init(4);
</script>
</body>
</html>
"""

html = html.replace("TREE_DATA_PLACEHOLDER", tree_data)
OUT_GINI.write_text(html, encoding="utf-8")
print(f"Interactive tree saved to: {OUT_GINI}")
