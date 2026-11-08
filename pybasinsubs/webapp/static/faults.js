// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// Fault editor: a canvas over the section image. The server (fault_tracer.FaultStore) owns the polylines, the undo
// history and the files; this file draws them and sends one operation per edit. Coordinates: x, z in metres
// (z positive down); extent = [xmin, xmax, zmax, zmin] as from faults.render_section_image.
"use strict";
(function () {
  const cv = $("f_cv"), ctx = cv.getContext("2d");
  const S = { st: null, mode: "draw", cur: [], sel: null, node: null, drag: null, bg: [], img: new Image(), ready: false };

  // ---- coordinate helpers (the JS version of the former apps/polyline.px_to_xz / xz_to_px)
  const ext = () => S.st.extent;
  const toPx = (x, z) => [(x - ext()[0]) / (ext()[1] - ext()[0]) * cv.width, (z - ext()[3]) / (ext()[2] - ext()[3]) * cv.height];
  const toXZ = (px, py) => [ext()[0] + px / cv.width * (ext()[1] - ext()[0]), ext()[3] + py / cv.height * (ext()[2] - ext()[3])];
  const colorOf = (id) => { const i = S.st.faults.findIndex((f) => f.id === id); return S.st.palette[(i < 0 ? 0 : i) % S.st.palette.length]; };

  function resize() {
    if (!S.st) return;
    const w = $("f_wrap").clientWidth || 900, ve = +$("f_ve").value;
    const h = Math.min(Math.max(w * (ext()[2] - ext()[3]) / (ext()[1] - ext()[0]) * ve, 200), 1000);
    cv.width = w; cv.height = Math.round(h); draw();
  }
  function line(pts, color, width, dash) {
    if (pts.length < 2) return;
    ctx.beginPath(); pts.forEach((p, i) => { const [a, b] = toPx(p[0], p[1]); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
  }
  function dots(pts, color, r, hot) {
    pts.forEach((p, i) => { const [a, b] = toPx(p[0], p[1]); ctx.beginPath(); ctx.arc(a, b, i === hot ? r + 2 : r, 0, 7);
      ctx.fillStyle = i === hot ? "#000" : color; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke(); });
  }
  function draw() {
    if (!S.st) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (S.ready) { ctx.globalAlpha = 0.8; ctx.drawImage(S.img, 0, 0, cv.width, cv.height); ctx.globalAlpha = 1; }
    S.bg.forEach((f) => line(f.nodes, "rgba(0,0,0,.6)", 1.5, [6, 4]));
    S.st.faults.forEach((f) => { line(f.nodes, colorOf(f.id), f.id === S.sel ? 5 : 3);
      const [a, b] = toPx(f.nodes[0][0], f.nodes[0][1]); ctx.fillStyle = colorOf(f.id); ctx.font = "12px sans-serif"; ctx.fillText(f.id, a + 5, b - 5); });
    const sel = S.st.faults.find((f) => f.id === S.sel);
    if (sel && S.mode === "select") dots(sel.nodes, colorOf(sel.id), 5, S.node);
    if (S.cur.length) { line(S.cur, "#000", 2, [4, 3]); dots(S.cur, "#000", 4, -1); }
  }

  // ---- state from the server
  function apply(st) {
    S.st = st;
    if (!st.faults.some((f) => f.id === S.sel)) { S.sel = st.faults.length ? st.faults[0].id : null; S.node = null; }
    $("f_fault").innerHTML = st.faults.map((f) => `<option>${f.id}</option>`).join(""); if (S.sel) $("f_fault").value = S.sel;
    $("f_tip").innerHTML = ["default"].concat(st.units).map((u) => `<option>${u}</option>`).join("");
    syncSel();
    $("f_undo").disabled = !st.can_undo; $("f_redo").disabled = !st.can_redo;
    $("f_ve").value = st.ve; $("f_ve_v").textContent = st.ve;
    if (st.message) setStatus("f_status", st.message);
    resize();
  }
  function syncSel() {
    const f = S.st.faults.find((x) => x.id === S.sel); $("f_name").value = f ? f.id : "";
    $("f_tip").value = f && f.tip !== null && f.tip !== undefined ? S.st.units[f.tip] : "default";
  }
  async function op(body) {
    try { apply(await apiJson("/api/faults/op", body)); } catch (e) { setStatus("f_status", "Error: " + e.message, true); }
  }
  async function post(path, body) {
    try { apply(await apiJson(path, body || {})); } catch (e) { setStatus("f_status", "Error: " + e.message, true); }
  }
  async function loadBg() {
    const k = $("f_bg").value;
    if (k === "none") { S.bg = []; draw(); return; }
    try { S.bg = (await apiJson("/api/faults/auto/" + k)).faults; } catch (e) { S.bg = []; }
    draw();
  }

  // ---- mouse
  function pos(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height]; }
  function distSeg(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)) : 0;
    return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
  }
  function pick(p) {       // nearest fault (by polyline) and, for the selected one, nearest node
    let best = null, bd = 9;
    S.st.faults.forEach((f) => { const px = f.nodes.map((n) => toPx(n[0], n[1]));
      for (let i = 0; i + 1 < px.length; i++) { const d = distSeg(p, px[i], px[i + 1]); if (d < bd) { bd = d; best = f.id; } } });
    return best;
  }
  function pickNode(p) {
    const f = S.st.faults.find((x) => x.id === S.sel); if (!f) return null;
    let best = null, bd = 9;
    f.nodes.forEach((n, i) => { const q = toPx(n[0], n[1]), d = Math.hypot(p[0] - q[0], p[1] - q[1]); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  cv.addEventListener("mousedown", (e) => {
    if (!S.st) return; const p = pos(e); cv.focus();
    if (S.mode === "draw") { if (e.detail < 2) S.cur.push(toXZ(p[0], p[1])); else finish(); draw(); return; }
    const n = pickNode(p);
    if (n !== null) { S.node = n; S.drag = { id: S.sel, j: n, moved: false }; draw(); return; }
    const id = pick(p); S.sel = id; S.node = null; if (id) { $("f_fault").value = id; syncSel(); } draw();
  });
  window.addEventListener("mousemove", (e) => {
    if (!S.drag) return; const f = S.st.faults.find((x) => x.id === S.drag.id), p = pos(e);
    f.nodes[S.drag.j] = toXZ(Math.max(0, Math.min(cv.width, p[0])), Math.max(0, Math.min(cv.height, p[1]))); S.drag.moved = true; draw();
  });
  window.addEventListener("mouseup", () => {
    if (!S.drag) return; const d = S.drag; S.drag = null;
    if (d.moved) op({ op: "move_node", id: d.id, j: d.j, xz: S.st.faults.find((x) => x.id === d.id).nodes[d.j] });
  });
  cv.addEventListener("keydown", (e) => {
    if (e.key === "Enter") finish();
    else if (e.key === "Escape") { S.cur = []; draw(); }
    else if (e.key === "Backspace" && S.mode === "draw") { S.cur.pop(); draw(); e.preventDefault(); }
    else if ((e.key === "Delete" || e.key === "Backspace") && S.mode === "select") delNode();
  });
  function finish() {
    if (S.cur.length < 2) { toast("a fault needs at least 2 nodes"); S.cur = []; draw(); return; }
    const nodes = S.cur; S.cur = []; op({ op: "add", nodes });
  }
  function delNode() {
    if (S.sel === null || S.node === null) { toast("select a node first (Select mode: click the fault, then a node)"); return; }
    op({ op: "delete_node", id: S.sel, j: S.node }); S.node = null;
  }
  function setMode(m) { S.mode = m; S.cur = []; $("f_draw").classList.toggle("on", m === "draw"); $("f_select").classList.toggle("on", m === "select"); draw(); }

  // ---- buttons
  $("f_draw").addEventListener("click", () => setMode("draw")); $("f_select").addEventListener("click", () => setMode("select"));
  $("f_finish").addEventListener("click", finish);
  $("f_fault").addEventListener("change", () => { S.sel = $("f_fault").value; S.node = null; syncSel(); draw(); });
  $("f_rename").addEventListener("click", () => {
    if (!S.sel) return;
    const name = $("f_name").value.trim();
    op({ op: "rename", id: S.sel, name }).then(() => {
      if (S.st.faults.some((f) => f.id === name)) { S.sel = name; S.node = null; apply(S.st); }
    });
  });
  $("f_tip").addEventListener("change", () => S.sel && op({ op: "set_tip", id: S.sel, tip: $("f_tip").value === "default" ? null : S.st.units.indexOf($("f_tip").value) }));
  $("f_ins").addEventListener("click", () => S.sel && op({ op: "insert_node", id: S.sel, j: S.node }));
  $("f_dn").addEventListener("click", delNode);
  $("f_del").addEventListener("click", () => S.sel && op({ op: "delete", id: S.sel }));
  $("f_undo").addEventListener("click", () => op({ op: "undo" })); $("f_redo").addEventListener("click", () => op({ op: "redo" }));
  $("f_resetb").addEventListener("click", () => op({ op: "reset", kind: $("f_reset").value }));
  $("f_save").addEventListener("click", () => post("/api/faults/save", { fmt: $("f_fmt").value }));
  $("f_load").addEventListener("click", () => post("/api/faults/load", { fmt: $("f_fmt").value }));
  $("f_dl").addEventListener("click", () => { location.href = "/api/faults/download/" + $("f_fmt").value; });
  $("f_up").addEventListener("change", async () => {
    const f = $("f_up").files[0]; if (!f) return; await post("/api/faults/import", { name: f.name, text: await f.text() }); $("f_up").value = "";
  });
  $("f_bg").addEventListener("change", loadBg);
  $("f_ve").addEventListener("input", () => { $("f_ve_v").textContent = $("f_ve").value; resize(); });
  $("f_ve").addEventListener("change", () => op({ op: "set_ve", value: +$("f_ve").value }));
  $("f_run").addEventListener("click", () => busy($("f_run"), async () => {
    setStatus("f_status", "Running restoration...");
    try {
      const r = await apiJson("/api/faults/restore", { method: $("f_method").value, Te: +$("f_te").value, Hd_km: +$("f_hd").value });
      setStatus("f_status", r.message); $("f_fig").src = "data:image/png;base64," + r.figure; $("f_fig").hidden = false;
    } catch (e) { setStatus("f_status", "Error: " + e.message, true); }
  }));
  window.addEventListener("resize", resize);

  PB.onShow.faults = resize;
  PB.initFaults = async function () {
    $("f_reset").innerHTML = PB.opts.resets.map((r) => `<option value="${r[1]}">${r[0]}</option>`).join("");
    S.img.onload = () => { S.ready = true; draw(); }; S.img.src = "/api/faults/image.png";
    try { const st = await apiJson("/api/faults/state"); st.message = `Loaded ${st.faults.length} faults (${st.stored ? "user file " + st.stored : "auto curved mid-ramp"}).`; apply(st); }
    catch (e) { setStatus("f_status", "Error: " + e.message, true); }
    loadBg();
  };
})();
