const state = {
  data: null,
  model: "16",
  membership: "primary",
  scope: "All tracks",
  year: "All years",
  selectedId: null,
  genderView: "lead",
  schoolView: "index",
};

const NS = "http://www.w3.org/2000/svg";
const TRACKS = ["Open", "Social Impact", "Health & Life Sciences"];
const COLORS = [
  "#cf3f52", "#4d8795", "#c59639", "#7a78a8", "#58a58f", "#bd6b3f", "#6d91c5", "#9c5f84",
  "#d37882", "#377d68", "#d0a961", "#6c8190", "#af4f62", "#4b9aa4", "#98753b", "#8d6d9e",
];
const fmt = new Intl.NumberFormat("en-US");
const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const number = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const signedPts = value => value == null ? "—" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pts`;
const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));

function model() { return state.data.models[state.model]; }
function slice(scope = state.scope, year = state.year, membership = state.membership) {
  return model().slices[membership][scope][year];
}
function concepts() { return model().concepts; }
function concept(id) { return concepts().find(item => item.id === Number(id)); }
function profile(id = state.selectedId) { return model().profiles[String(id)]; }
function color(id) { return COLORS[Number(id) % COLORS.length]; }
function svgEl(name, attrs = {}) {
  const element = document.createElementNS(NS, name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}
function scale(value, inMin, inMax, outMin, outMax) {
  if (inMax === inMin) return (outMin + outMax) / 2;
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

function ensureSelected() {
  const available = slice().items;
  const valid = concepts().some(item => item.id === state.selectedId);
  if (!valid) state.selectedId = available[0]?.id ?? concepts()[0]?.id ?? null;
}

function currentOutcome() {
  const inSlice = slice().items.find(item => item.id === state.selectedId);
  if (inSlice) return { outcome: inSlice, fallback: false };
  const data = profile();
  const fallback = state.membership === "active" ? data?.activeOutcome : data?.primaryOutcome;
  return { outcome: fallback || data?.activeOutcome || null, fallback: true };
}

function selectConcept(id, scroll = false) {
  state.selectedId = Number(id);
  renderRibbon();
  renderLandscape();
  renderProfile();
  renderJudges();
  if (scroll) document.querySelector("#profile").scrollIntoView({ behavior: "smooth", block: "start" });
}

function selectLeadershipConcept(id) {
  state.model = "16";
  state.membership = "primary";
  state.scope = "All tracks";
  state.year = "All years";
  state.selectedId = Number(id);
  renderAll();
  document.querySelector("#profile").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHeader() {
  const { meta } = state.data;
  document.querySelector("#hero-judge-rows").textContent = fmt.format(meta.judgeRows);
  document.querySelector("#hero-applications").textContent = fmt.format(meta.applications);
  document.querySelector("#hero-eligible").textContent = fmt.format(meta.eligibleProblemTexts);
}

function renderControls() {
  document.querySelectorAll("#model-control button").forEach(button => button.classList.toggle("active", button.dataset.model === state.model));
  document.querySelectorAll("#membership-control button").forEach(button => button.classList.toggle("active", button.dataset.membership === state.membership));
  document.querySelector("#scope-control").value = state.scope;
  document.querySelector("#year-control").value = state.year;
  document.querySelector("#control-visible").textContent = slice().items.length;
  document.querySelectorAll("#gender-view-control button").forEach(button => button.classList.toggle("active", button.dataset.genderView === state.genderView));
  document.querySelectorAll("#school-view-control button").forEach(button => button.classList.toggle("active", button.dataset.schoolView === state.schoolView));
}

function renderRibbon() {
  const data = slice();
  const holder = document.querySelector("#concept-ribbon");
  const legend = document.querySelector("#ribbon-legend");
  const total = state.membership === "primary" ? Math.max(data.scopeN, 1) : Math.max(data.visibleN, 1);
  document.querySelector("#ribbon-title").textContent = state.membership === "primary" ? "Primary portfolio share" : "Overlapping concept memberships";
  document.querySelector("#ribbon-note").textContent = state.membership === "primary"
    ? "Width shows each displayed area’s share of applications; striped space combines suppressed small areas."
    : "Width shows each concept’s share of displayed feature activations. Applications can contribute to several concepts.";
  holder.innerHTML = "";
  data.items.forEach(item => {
    const segment = document.createElement("button");
    segment.type = "button";
    segment.className = `ribbon-segment ${item.id === state.selectedId ? "active" : ""}`;
    segment.style.width = `${(item.n / total) * 100}%`;
    segment.style.background = color(item.id);
    segment.title = `${item.label}: ${item.n}`;
    segment.setAttribute("aria-label", `${item.label}, ${item.n} applications`);
    if ((item.n / total) > .055) segment.innerHTML = `<span>${item.id + 1}</span>`;
    segment.addEventListener("click", () => selectConcept(item.id));
    holder.append(segment);
  });
  if (state.membership === "primary" && data.suppressedN > 0) {
    const suppressed = document.createElement("div");
    suppressed.className = "ribbon-segment suppressed";
    suppressed.style.width = `${(data.suppressedN / total) * 100}%`;
    suppressed.title = `${data.suppressedN} applications across suppressed areas`;
    holder.append(suppressed);
  }
  legend.innerHTML = data.items.slice(0, 10).map(item => `<span class="ribbon-key"><i style="background:${color(item.id)}"></i>${esc(item.id + 1)} · ${esc(item.label)}</span>`).join("")
    + (data.items.length > 10 ? `<span class="ribbon-key">+ ${data.items.length - 10} more displayed</span>` : "");
}

function renderFindings() {
  const cards = state.data.findings.slice(0, 4);
  document.querySelector("#finding-grid").innerHTML = cards.map(card => `<article class="finding-card">
    <small>Signal</small><h3>${esc(card.signal)}</h3><strong>${esc(card.value)}</strong><h4>${esc(card.title)}</h4>
    <dl><dt>Evidence</dt><dd>${esc(card.evidence)}</dd><dt>Possible i-lab implication</dt><dd>${esc(card.implication)}</dd></dl>
  </article>`).join("");
}

function signalRows(items, formatter) {
  return items.map(item => `<button type="button" class="signal-row" data-id="${item.id}"><span>${esc(item.label)}</span><b>${formatter(item)}</b><small>N=${item.n} applications</small></button>`).join("") || `<p class="panel-note">No publishable entry meets the criteria.</p>`;
}

function renderActionable() {
  const signals = state.data.actionableSignals;
  document.querySelector("#action-evaluation").innerHTML = signalRows(signals.strongerEvaluationLowerAttention, item => `${item.adjustedRecommendation >= 0 ? "+" : ""}${number(item.adjustedRecommendation)} adjusted`);
  document.querySelector("#action-support").innerHTML = signalRows(signals.founderSupportGaps, item => `Problem − business ${item.problemMinusBusiness >= 0 ? "+" : ""}${number(item.problemMinusBusiness)}`);
  document.querySelector("#action-disagreement").innerHTML = signalRows(signals.judgeDisagreement, item => `Within-application SD ${number(item.meanWithinApplicationSd)}`);
  document.querySelectorAll(".signal-row[data-id]").forEach(row => row.addEventListener("click", () => selectLeadershipConcept(row.dataset.id)));
}

function renderSupportPortfolio() {
  const rows = state.data.models["16"].supportPortfolio;
  const metrics = [["problem", "Problem"], ["solution", "Prototype"], ["business", "Business"], ["impact", "Impact"]];
  document.querySelector("#support-portfolio").innerHTML = rows.map(item => `<button type="button" class="support-row" data-id="${item.id}">
    <div class="support-label"><strong>${esc(item.label)}</strong><span>${esc(item.pattern)} · N=${item.n} applications</span></div>
    <div class="support-scores">${metrics.map(([key, label]) => `<div title="${label}: ${number(item[key])}"><i class="score-${key}" style="width:${item[key] / 5 * 100}%"></i><span>${number(item[key])}</span></div>`).join("")}</div>
    <div class="support-gap"><span>Problem − business</span><b>${item.problemMinusBusiness >= 0 ? "+" : ""}${number(item.problemMinusBusiness)}</b></div>
  </button>`).join("");
  document.querySelectorAll("#support-portfolio [data-id]").forEach(row => row.addEventListener("click", () => selectLeadershipConcept(row.dataset.id)));
}

function attachBubbleEvents(circle, item, tooltipText) {
  const tooltip = document.querySelector("#tooltip");
  const show = event => {
    tooltip.hidden = false;
    tooltip.innerHTML = `<strong>${esc(item.label)}</strong>${tooltipText}`;
    const box = circle.getBoundingClientRect();
    const x = event.clientX || box.left + box.width;
    const y = event.clientY || box.top;
    tooltip.style.left = `${Math.min(x + 14, window.innerWidth - 290)}px`;
    tooltip.style.top = `${Math.max(y - 30, 10)}px`;
  };
  circle.addEventListener("mousemove", show);
  circle.addEventListener("focus", show);
  circle.addEventListener("mouseleave", () => tooltip.hidden = true);
  circle.addEventListener("blur", () => tooltip.hidden = true);
  circle.addEventListener("click", () => selectConcept(item.id));
  circle.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") selectConcept(item.id);
  });
}

function renderLandscapeChart() {
  const items = slice().items;
  const container = document.querySelector("#landscape-chart");
  container.innerHTML = "";
  if (!items.length) {
    container.innerHTML = `<div class="empty-state">No cells meet the public N ≥ 10 threshold for this combination.<br />Choose all years, a broader track, or overlapping concepts.</div>`;
    return;
  }
  const useTrend = state.membership === "primary" && state.year === "All years";
  const width = 1160, height = 520;
  const margin = { top: 27, right: 45, bottom: 60, left: 70 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xValues = items.map(item => item.adjustedRecommendation).filter(value => value != null);
  const xExtent = Math.max(.15, ...xValues.map(Math.abs)) * 1.15;
  const xMin = -xExtent, xMax = xExtent;
  const yValues = items.map(item => useTrend ? item.trend : item.share).filter(value => value != null);
  const extent = useTrend ? Math.ceil(Math.max(.02, ...yValues.map(Math.abs)) / .02) * .02 : Math.max(.1, ...yValues) * 1.12;
  const yMin = useTrend ? -extent : 0;
  const yMax = extent;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" });
  [-xExtent, -xExtent / 2, 0, xExtent / 2, xExtent].forEach(tick => {
    const x = scale(tick, xMin, xMax, margin.left, margin.left + plotW);
    svg.append(svgEl("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + plotH, class: "grid-line" }));
    const label = svgEl("text", { x, y: height - 27, "text-anchor": "middle", class: "axis-label" });
    label.textContent = `${tick > 0 ? "+" : ""}${tick.toFixed(2)}`; svg.append(label);
  });
  const yTicks = useTrend ? [-extent, -extent / 2, 0, extent / 2, extent] : [0, yMax / 4, yMax / 2, yMax * .75, yMax];
  yTicks.forEach(tick => {
    const y = scale(tick, yMin, yMax, margin.top + plotH, margin.top);
    svg.append(svgEl("line", { x1: margin.left, y1: y, x2: margin.left + plotW, y2: y, class: tick === 0 ? "grid-line zero-line" : "grid-line" }));
    const label = svgEl("text", { x: margin.left - 11, y: y + 3, "text-anchor": "end", class: "axis-label" });
    label.textContent = useTrend ? `${tick > 0 ? "+" : ""}${(tick * 100).toFixed(1).replace(".0", "")} pts` : `${(tick * 100).toFixed(0)}%`;
    svg.append(label);
  });
  const xTitle = svgEl("text", { x: margin.left + plotW / 2, y: height - 4, "text-anchor": "middle", class: "axis-title" });
  xTitle.textContent = "Year×track-adjusted application Recommendation →"; svg.append(xTitle);
  const yTitle = svgEl("text", { x: 14, y: margin.top + plotH / 2, transform: `rotate(-90 14 ${margin.top + plotH / 2})`, "text-anchor": "middle", class: "axis-title" });
  yTitle.textContent = useTrend ? "2022–2024 yearly change in portfolio share →" : "Share of applications in selected scope →"; svg.append(yTitle);
  const maxN = Math.max(...items.map(item => item.n));
  items.forEach(item => {
    const yValue = useTrend ? item.trend : item.share;
    if (item.adjustedRecommendation == null || yValue == null) return;
    const x = clamp(scale(item.adjustedRecommendation, xMin, xMax, margin.left, margin.left + plotW), margin.left, margin.left + plotW);
    const y = scale(yValue, yMin, yMax, margin.top + plotH, margin.top);
    const radius = scale(Math.sqrt(item.n), Math.sqrt(10), Math.sqrt(maxN), 13, 38);
    const circle = svgEl("circle", { cx: x, cy: y, r: radius, fill: color(item.id), "fill-opacity": ".82", class: `bubble ${item.id === state.selectedId ? "selected" : ""}`, tabindex: "0" });
    attachBubbleEvents(circle, item, `${item.n} applications · raw ${number(item.recommendation)}/5 · adjusted ${item.adjustedRecommendation >= 0 ? "+" : ""}${number(item.adjustedRecommendation)} · ${useTrend ? signedPts(item.trend2022To2024) + "/year, 2022–2024" : pct(item.share) + " of scope"}`);
    svg.append(circle);
    if (item.n >= Math.max(15, maxN * .23)) {
      const label = svgEl("text", { x, y: y + radius + 14, "text-anchor": "middle", class: "bubble-label" });
      label.textContent = `${item.id + 1} · ${item.label.length > 28 ? item.label.slice(0, 27) + "…" : item.label}`;
      svg.append(label);
    }
  });
  container.append(svg);
}

function renderLandscapeTable() {
  const items = slice().items;
  const useTrend = state.membership === "primary" && state.year === "All years";
  document.querySelector("#table-change-heading").textContent = useTrend ? "2022–2024 trend / year" : "Share of scope";
  document.querySelector("#landscape-table tbody").innerHTML = items.map(item => `<tr data-id="${item.id}">
    <td>${esc(item.id + 1)} · ${esc(item.label)}</td><td>${fmt.format(item.n)}</td><td>${pct(item.share)}</td>
    <td>${number(item.recommendation)} <small>(${number(item.recommendationCiLow)}–${number(item.recommendationCiHigh)})</small></td>
    <td>${item.adjustedRecommendation >= 0 ? "+" : ""}${number(item.adjustedRecommendation)} <small>(${number(item.adjustedRecommendationCiLow)}–${number(item.adjustedRecommendationCiHigh)})</small></td>
    <td>${item.ratings == null ? "—" : fmt.format(item.ratings)}</td><td>${useTrend ? signedPts(item.trend2022To2024) : pct(item.share)}</td>
  </tr>`).join("") || `<tr><td colspan="7">No publishable cells for this selection.</td></tr>`;
  document.querySelectorAll("#landscape-table tbody tr[data-id]").forEach(row => row.addEventListener("click", () => selectConcept(row.dataset.id, true)));
}

function renderLandscape() {
  const useTrend = state.membership === "primary" && state.year === "All years";
  document.querySelector("#landscape-description").textContent = useTrend
    ? "Bubble size shows application count. Position compares year×track-adjusted Recommendation with the 2022–2024 change in entrepreneurial attention."
    : "Bubble size shows application count. Position compares year×track-adjusted Recommendation with prevalence in the selected scope.";
  document.querySelector("#chart-scope-label").textContent = `${state.scope} · ${state.year} · ${state.membership === "primary" ? "primary assignment" : "overlapping membership"}`;
  renderLandscapeChart();
  renderLandscapeTable();
}

function renderScoreBars(outcome) {
  const metrics = [
    ["Problem / customer", outcome?.problem], ["Solution / prototype", outcome?.solution],
    ["Business model", outcome?.business], ["Impact / potential", outcome?.impact],
  ];
  document.querySelector("#score-bars").innerHTML = metrics.map(([label, value]) => `<div class="score-row"><span>${esc(label)}</span><div class="score-track"><div class="score-fill" style="width:${value == null ? 0 : value / 5 * 100}%"></div></div><strong>${number(value)}</strong></div>`).join("");
  const valid = metrics.filter(([, value]) => value != null);
  if (!valid.length) return "No publishable judging profile is available for this selection.";
  const strongest = [...valid].sort((a, b) => b[1] - a[1])[0];
  const weakest = [...valid].sort((a, b) => a[1] - b[1])[0];
  return `${strongest[0]} is strongest on average (${number(strongest[1])}); ${weakest[0].toLowerCase()} is weakest (${number(weakest[1])}). Use this as a programming question, not a causal conclusion.`;
}

function renderTrend(profileData) {
  const holder = document.querySelector("#trend-chart");
  holder.innerHTML = "";
  const trend = profileData?.trend;
  document.querySelector("#trend-value").textContent = trend ? `${signedPts(trend.slope2022To2024)} / yr` : "Suppressed";
  if (!trend) {
    holder.innerHTML = `<div class="empty-state">Primary-area N is below the public threshold.</div>`;
    return;
  }
  const width = 500, height = 180, left = 45, right = 15, top = 20, bottom = 32;
  const values = trend.years.filter(item => item.share != null).map(item => item.share);
  const max = Math.max(.1, ...values) * 1.1;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" });
  [0, max / 2, max].forEach(tick => {
    const y = scale(tick, 0, max, height - bottom, top);
    svg.append(svgEl("line", { x1: left, y1: y, x2: width - right, y2: y, stroke: "#dedbd3", "stroke-width": 1 }));
    const label = svgEl("text", { x: left - 8, y: y + 3, "text-anchor": "end", fill: "#78858a", "font-size": 9, "font-family": "DM Mono" });
    label.textContent = `${(tick * 100).toFixed(0)}%`; svg.append(label);
  });
  const comparablePoints = [];
  let partialPoint = null;
  trend.years.forEach((item, index) => {
    const x = scale(index, 0, 3, left, width - right);
    const year = svgEl("text", { x, y: height - 9, "text-anchor": "middle", fill: "#78858a", "font-size": 9, "font-family": "DM Mono" });
    year.textContent = item.year === 2021 ? "2021*" : item.year; svg.append(year);
    if (item.share == null) {
      const mark = svgEl("text", { x, y: height / 2, "text-anchor": "middle", fill: "#a51c30", "font-size": 9, "font-family": "DM Mono" });
      mark.textContent = "<10"; svg.append(mark);
    } else {
      const point = [x, scale(item.share, 0, max, height - bottom, top)];
      if (item.year === 2021) partialPoint = point; else comparablePoints.push(point);
    }
  });
  if (partialPoint && comparablePoints.length) svg.append(svgEl("line", { x1: partialPoint[0], y1: partialPoint[1], x2: comparablePoints[0][0], y2: comparablePoints[0][1], stroke: "#a51c30", "stroke-width": 2, "stroke-dasharray": "4 5", opacity: .55 }));
  if (comparablePoints.length > 1) svg.append(svgEl("polyline", { points: comparablePoints.map(point => point.join(",")).join(" "), fill: "none", stroke: "#a51c30", "stroke-width": 3 }));
  comparablePoints.forEach(([x, y]) => svg.append(svgEl("circle", { cx: x, cy: y, r: 5, fill: "#a51c30", stroke: "#fff", "stroke-width": 2 })));
  if (partialPoint) svg.append(svgEl("circle", { cx: partialPoint[0], cy: partialPoint[1], r: 5, fill: "#fff", stroke: "#a51c30", "stroke-width": 2 }));
  holder.append(svg);
  document.querySelector("#trend-note").textContent = `Primary 2022–2024 change: ${signedPts(trend.change2022To2024)}; slope ${signedPts(trend.slope2022To2024)}/year. Research detail: 2021–2024 partial-coverage slope ${signedPts(trend.slope2021To2024Partial)}/year. Exact yearly cells below 10 applications are hidden; 2021* is not fully comparable.`;
}

function renderCompositions(profileData) {
  const labels = { tracks: "Track", leadGender: "Lead applicant gender", schools: "Harvard school", countries: "Country", industries: "Industry", years: "Year" };
  document.querySelector("#composition-grid").innerHTML = Object.entries(labels).map(([key, heading]) => {
    const composition = profileData.compositions[key];
    const rows = composition.values.length
      ? composition.values.slice(0, 7).map(item => `<li><span>${esc(item.label)}</span><b>${fmt.format(item.n)}</b></li>`).join("")
      : `<li><span>No cell meets N ≥ 10</span><b>—</b></li>`;
    return `<div class="composition-block"><h4>${esc(heading)}</h4><ul>${rows}</ul>${composition.suppressedCells ? `<small>${composition.suppressedCells} smaller categor${composition.suppressedCells === 1 ? "y" : "ies"} suppressed</small>` : ""}</div>`;
  }).join("") + `<div class="composition-block"><h4>Track overrepresentation</h4><ul>${profileData.trackOverrepresentation.length ? profileData.trackOverrepresentation.map(item => `<li><span>${esc(item.track)}</span><b>${number(item.ratio)}× · N=${item.n}</b></li>`).join("") : `<li><span>No cell meets N ≥ 10</span><b>—</b></li>`}</ul></div>`
    + `<div class="composition-block"><h4>School representation index</h4><ul>${profileData.schoolRepresentation.length ? profileData.schoolRepresentation.slice(0, 6).map(item => `<li><span>${esc(item.school)}</span><b>${number(item.representationIndex)}× · N=${item.n}</b></li>`).join("") : `<li><span>No cell meets N ≥ 10</span><b>—</b></li>`}</ul></div>`;
}

function renderProfile() {
  ensureSelected();
  const meta = concept(state.selectedId);
  const data = profile();
  const { outcome, fallback } = currentOutcome();
  const select = document.querySelector("#concept-control");
  select.innerHTML = concepts().map(item => `<option value="${item.id}">${state.model}.${String(item.id + 1).padStart(2, "0")} · ${esc(item.label)}${item.primarySuppressed ? " · primary N<10" : ""}</option>`).join("");
  select.value = String(state.selectedId);
  document.querySelector("#profile-code").textContent = `M=${state.model} · CONCEPT ${String(meta.id + 1).padStart(2, "0")}`;
  document.querySelector("#profile-title").textContent = meta.label;
  document.querySelector("#profile-description").textContent = meta.description;
  document.querySelector("#profile-quality-note").textContent = `Concept review: ${meta.qualityNote}`;
  const badges = [
    meta.qualityBadge,
    `Seed stability: ${meta.stability}`,
    `Membership Jaccard: ${number(meta.membershipJaccard)}`,
    `Active in ${pct(meta.activePrevalence)}`,
    meta.primarySuppressed ? "Primary N < 10" : `Primary N ${meta.primaryN}`,
    fallback ? "Filtered cell suppressed · showing all-track profile" : `${state.scope} · ${state.year}`,
  ];
  document.querySelector("#profile-badges").innerHTML = badges.map(label => `<span class="badge">${esc(label)}</span>`).join("");
  const disagreement = (model().disagreement[state.scope] || []).find(item => item.id === meta.id)
    || model().disagreement["All tracks"].find(item => item.id === meta.id);
  const metrics = [
    [state.membership === "primary" ? "Applications in primary area" : "Active memberships", outcome?.n == null ? "Suppressed" : fmt.format(outcome.n)],
    ["Portfolio share", pct(outcome?.share)],
    ["Raw Recommendation", outcome?.recommendation == null ? "—" : `${number(outcome.recommendation)} (${number(outcome.recommendationCiLow)}–${number(outcome.recommendationCiHigh)})`],
    ["Year×track adjusted", outcome?.adjustedRecommendation == null ? "—" : `${outcome.adjustedRecommendation >= 0 ? "+" : ""}${number(outcome.adjustedRecommendation)} (${number(outcome.adjustedRecommendationCiLow)}–${number(outcome.adjustedRecommendationCiHigh)})`],
    ["Judge ratings", outcome?.ratings == null ? "—" : fmt.format(outcome.ratings)],
    ["Judge disagreement", number(disagreement?.meanWithinApplicationSd)],
  ];
  document.querySelector("#profile-metrics").innerHTML = metrics.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("");
  document.querySelector("#support-pattern").textContent = data.support?.pattern || "Primary profile suppressed";
  const scoreInterpretation = renderScoreBars(outcome);
  document.querySelector("#support-note").textContent = data.support
    ? `${scoreInterpretation} Problem − business: ${data.support.problemMinusBusiness >= 0 ? "+" : ""}${number(data.support.problemMinusBusiness)}; problem − prototype: ${data.support.problemMinusPrototype >= 0 ? "+" : ""}${number(data.support.problemMinusPrototype)}; impact − business: ${data.support.impactMinusBusiness >= 0 ? "+" : ""}${number(data.support.impactMinusBusiness)}. Historical profile only; no causal interpretation.`
    : scoreInterpretation;
  renderTrend(data);
  renderCompositions(data);
  document.querySelector("#related-list").innerHTML = data.related.length
    ? data.related.map(item => `<button type="button" class="related-item" data-id="${item.id}"><strong>${esc(item.label)}</strong><span>N overlap ${item.nOverlap}<br />Jaccard ${number(item.jaccard)}</span></button>`).join("")
    : `<p class="panel-note">No overlap cell meets N ≥ 10.</p>`;
  document.querySelectorAll("#related-list [data-id]").forEach(button => button.addEventListener("click", () => selectConcept(button.dataset.id)));
  document.querySelector("#judge-type-list").innerHTML = data.judgeTypes.length
    ? data.judgeTypes.map(item => `<div class="judge-type-row"><strong>${esc(item.label)}</strong><span>${number(item.mean)} ± ${number(item.sd)}</span><span>N=${item.ratings}</span></div>`).join("")
    : `<p class="panel-note">No judge-type cell meets the public threshold.</p>`;
}

function renderTracks() {
  const selectedYear = state.year;
  const trackData = Object.fromEntries(TRACKS.map(track => [track, slice(track, selectedYear)]));
  document.querySelector("#track-cards").innerHTML = TRACKS.map(track => {
    const data = trackData[track];
    const top = data.items.slice(0, 5);
    const max = Math.max(1, ...top.map(item => item.share || 0));
    const signatures = model().trackSignatures[track] || [];
    const signatureHtml = state.membership === "primary" && state.year === "All years" && signatures.length
      ? `<div class="signature-label">Track signature · relative concentration</div>${signatures.slice(0, 3).map(item => `<div class="track-rank signature" data-id="${item.id}"><div class="track-rank-head"><span>${esc(item.label)}</span><b>${number(item.ratio)}× · N=${item.n}</b></div><div class="rank-track"><div class="rank-fill" style="width:${Math.min(100, item.ratio / Math.max(...signatures.map(row => row.ratio)) * 100)}%"></div></div></div>`).join("")}`
      : `<div class="signature-label">Current-filter portfolio share</div>${top.map(item => `<div class="track-rank" data-id="${item.id}"><div class="track-rank-head"><span>${esc(item.label)}</span><b>${pct(item.share)}</b></div><div class="rank-track"><div class="rank-fill" style="width:${(item.share / max) * 100}%"></div></div></div>`).join("")}`;
    return `<article class="track-card"><header><h3>${esc(track)}</h3><span>${fmt.format(data.scopeN)} applications</span></header><p>${data.items.length} problem areas meet N ≥ 10 · ${state.membership === "primary" ? `${data.suppressedN} applications in smaller areas` : "overlapping memberships"}</p>${signatureHtml}</article>`;
  }).join("");
  document.querySelectorAll(".track-rank[data-id]").forEach(row => row.addEventListener("click", () => selectConcept(row.dataset.id, true)));
  const union = new Map();
  TRACKS.forEach(track => trackData[track].items.forEach(item => {
    if (!union.has(item.id)) union.set(item.id, { id: item.id, label: item.label, values: {} });
    union.get(item.id).values[track] = item;
  }));
  const rows = [...union.values()].sort((a, b) => Math.max(...Object.values(b.values).map(v => v.share)) - Math.max(...Object.values(a.values).map(v => v.share)));
  document.querySelector("#track-table tbody").innerHTML = rows.map(row => `<tr data-id="${row.id}"><td>${row.id + 1} · ${esc(row.label)}</td>${TRACKS.map(track => `<td>${row.values[track] ? `${pct(row.values[track].share)} · N=${row.values[track].n}` : "Suppressed"}</td>`).join("")}</tr>`).join("");
  document.querySelectorAll("#track-table tbody tr[data-id]").forEach(row => row.addEventListener("click", () => selectConcept(row.dataset.id, true)));
}

function groupedCells(kind, valueKey = "n") {
  const grouped = new Map();
  model().founderMatrix[kind].forEach(cell => {
    if (!grouped.has(cell.conceptId)) grouped.set(cell.conceptId, { id: cell.conceptId, label: cell.conceptLabel, values: {} });
    grouped.get(cell.conceptId).values[cell.category] = { value: cell[valueKey], n: cell.n };
  });
  return [...grouped.values()];
}

function renderFounderLandscape() {
  const genderKind = state.genderView === "lead" ? "leadGender" : "teamGender";
  const genderRows = groupedCells(genderKind);
  const genderTotals = {};
  model().founderMatrix[genderKind].forEach(cell => genderTotals[cell.category] = (genderTotals[cell.category] || 0) + cell.n);
  const categories = Object.entries(genderTotals).sort((a, b) => b[1] - a[1]).slice(0, 3).map(entry => entry[0]);
  const visibleIds = new Set(slice("All tracks", "All years", "primary").items.map(item => item.id));
  const filteredGender = genderRows.filter(row => visibleIds.has(row.id)).sort((a, b) => Object.values(b.values).reduce((x, y) => x + y.n, 0) - Object.values(a.values).reduce((x, y) => x + y.n, 0));
  const maxGender = Math.max(1, ...filteredGender.flatMap(row => categories.map(category => row.values[category]?.n || 0)));
  document.querySelector("#gender-matrix").innerHTML = `<div class="matrix-row header"><div>Problem area</div>${categories.map(category => `<div>${esc(category)}</div>`).join("")}</div>` + filteredGender.map(row => `<div class="matrix-row"><div>${row.id + 1} · ${esc(row.label)}</div>${categories.map(category => {
    const cell = row.values[category];
    return `<div class="matrix-cell" style="--heat:${cell ? .12 + .7 * cell.n / maxGender : 0}">${cell ? `<i></i><span>${cell.n}</span>` : `<span>—</span>`}</div>`;
  }).join("")}</div>`).join("");
  const coverage = model().founderMatrix.genderCoverage[state.genderView];
  document.querySelector("#gender-coverage-note").textContent = state.genderView === "lead"
    ? `Default application-level Gender field: ${fmt.format(coverage.reported)} of ${fmt.format(coverage.total)} applications reported (${pct(coverage.coverage)}). Cells N<10 are suppressed.`
    : `Secondary team-member view: ${fmt.format(coverage.reported)} of ${fmt.format(coverage.total)} supplied lead/team records report gender (${pct(coverage.coverage)}). Missingness is substantial and cells N<10 are suppressed.`;

  const schoolKind = state.schoolView === "index" ? "schoolRepresentation" : "schoolCounts";
  const schoolValue = state.schoolView === "index" ? "representationIndex" : "n";
  const schoolRows = groupedCells(schoolKind, schoolValue).filter(row => visibleIds.has(row.id));
  const schoolTotals = {};
  model().founderMatrix[schoolKind].forEach(cell => schoolTotals[cell.category] = (schoolTotals[cell.category] || 0) + cell.n);
  const schools = Object.entries(schoolTotals).sort((a, b) => b[1] - a[1]).slice(0, 8).map(entry => entry[0]);
  const maxSchool = Math.max(1, ...schoolRows.flatMap(row => schools.map(school => row.values[school]?.value || 0)));
  document.querySelector("#school-matrix").innerHTML = `<table class="heatmap"><thead><tr><th>Problem area</th>${schools.map(school => `<th>${esc(school.replace("Harvard ", ""))}</th>`).join("")}</tr></thead><tbody>${schoolRows.sort((a, b) => a.id - b.id).map(row => `<tr><td>${row.id + 1} · ${esc(row.label)}</td>${schools.map(school => {
    const cell = row.values[school];
    const alpha = cell ? .1 + .72 * cell.value / maxSchool : 0;
    const display = cell ? (state.schoolView === "index" ? `${number(cell.value)}×` : cell.n) : "—";
    return `<td title="${cell ? `N=${cell.n}` : "Suppressed or absent"}" style="background:rgba(54,106,120,${alpha})">${display}</td>`;
  }).join("")}</tr>`).join("")}</tbody></table>`;

  [["countries", "#country-summary"], ["industries", "#industry-summary"]].forEach(([kind, selector]) => {
    const totals = new Map();
    model().founderMatrix[kind].forEach(cell => {
      if (!visibleIds.has(cell.conceptId)) return;
      const current = totals.get(cell.category) || { label: cell.category, n: 0, concepts: 0 };
      current.n += cell.n; current.concepts += 1; totals.set(cell.category, current);
    });
    document.querySelector(selector).innerHTML = [...totals.values()].sort((a, b) => b.n - a.n).slice(0, 10).map(item => `<div class="rank-item"><div><span>${esc(item.label)}</span><small>visible across ${item.concepts} problem areas</small></div><strong>${fmt.format(item.n)}</strong></div>`).join("") || `<p class="panel-note">No cells meet N ≥ 10.</p>`;
  });
}

function renderDisagreementChart(items) {
  const holder = document.querySelector("#disagreement-chart");
  holder.innerHTML = "";
  if (!items.length) { holder.innerHTML = `<div class="empty-state">No disagreement cells meet N ≥ 10.</div>`; return; }
  const width = 780, height = 420, margin = { top: 25, right: 30, bottom: 55, left: 64 };
  const plotW = width - margin.left - margin.right, plotH = height - margin.top - margin.bottom;
  const xMin = 2.5, xMax = 4.2;
  const yMin = Math.min(.75, ...items.map(item => item.meanWithinApplicationSd)) - .03;
  const yMax = Math.max(1.2, ...items.map(item => item.meanWithinApplicationSd)) + .03;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" });
  [2.6, 3.0, 3.4, 3.8, 4.2].forEach(tick => {
    const x = scale(tick, xMin, xMax, margin.left, margin.left + plotW);
    svg.append(svgEl("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + plotH, class: "grid-line" }));
    const label = svgEl("text", { x, y: height - 25, "text-anchor": "middle", class: "axis-label" }); label.textContent = tick.toFixed(1); svg.append(label);
  });
  [yMin, (yMin + yMax) / 2, yMax].forEach(tick => {
    const y = scale(tick, yMin, yMax, margin.top + plotH, margin.top);
    svg.append(svgEl("line", { x1: margin.left, y1: y, x2: margin.left + plotW, y2: y, class: "grid-line" }));
    const label = svgEl("text", { x: margin.left - 9, y: y + 3, "text-anchor": "end", class: "axis-label" }); label.textContent = tick.toFixed(2); svg.append(label);
  });
  const maxN = Math.max(...items.map(item => item.n));
  items.forEach(item => {
    const x = scale(item.recommendation, xMin, xMax, margin.left, margin.left + plotW);
    const y = scale(item.meanWithinApplicationSd, yMin, yMax, margin.top + plotH, margin.top);
    const r = scale(Math.sqrt(item.n), Math.sqrt(10), Math.sqrt(maxN), 10, 28);
    const circle = svgEl("circle", { cx: x, cy: y, r, fill: color(item.id), "fill-opacity": .82, class: `bubble ${item.id === state.selectedId ? "selected" : ""}`, tabindex: 0 });
    attachBubbleEvents(circle, item, `mean ${number(item.recommendation)} · within-application SD ${number(item.meanWithinApplicationSd)} · N=${item.n} applications`);
    svg.append(circle);
  });
  const xTitle = svgEl("text", { x: margin.left + plotW / 2, y: height - 3, "text-anchor": "middle", class: "axis-title" }); xTitle.textContent = "Mean application Recommendation →"; svg.append(xTitle);
  const yTitle = svgEl("text", { x: 14, y: margin.top + plotH / 2, transform: `rotate(-90 14 ${margin.top + plotH / 2})`, "text-anchor": "middle", class: "axis-title" }); yTitle.textContent = "Mean within-application Recommendation SD →"; svg.append(yTitle);
  holder.append(svg);
}

function renderJudges() {
  const items = model().disagreement[state.scope] || [];
  document.querySelector("#judge-scope-label").textContent = `${state.scope} · all years · primary areas`;
  renderDisagreementChart(items);
  document.querySelector("#polarizing-list").innerHTML = [...items].sort((a, b) => b.meanWithinApplicationSd - a.meanWithinApplicationSd).slice(0, 6).map(item => `<div class="polar-row" data-id="${item.id}"><strong>${item.id + 1} · ${esc(item.label)}</strong><div><span>Within-application SD ${number(item.meanWithinApplicationSd)}</span><span>N=${item.n} applications</span></div></div>`).join("") || `<p>No cells meet N ≥ 10.</p>`;
  document.querySelectorAll(".polar-row[data-id]").forEach(row => row.addEventListener("click", () => selectConcept(row.dataset.id, true)));
  const rows = profile()?.judgeTypes || [];
  document.querySelector("#judge-table tbody").innerHTML = rows.map(item => `<tr><td>${esc(item.label)}</td><td>${number(item.mean)}</td><td>${number(item.sd)}</td><td>${number(item.median)}</td><td>${item.ratings}</td><td>${item.applications}</td><td>${item.adjustedMean >= 0 ? "+" : ""}${number(item.adjustedMean, 3)}</td></tr>`).join("") || `<tr><td colspan="7">No judge-type cell meets N ≥ 10 for this concept.</td></tr>`;
}

function renderMethod() {
  const cards = Object.values(state.data.models).map(item => item.summary);
  document.querySelector("#model-cards").innerHTML = cards.map(summary => `<article class="model-card ${summary.m === 16 ? "recommended" : ""}"><header><h3>M=${summary.m} · K=${summary.k}</h3><span>${summary.m === 16 ? "Recommended leadership taxonomy" : "Detailed / exploratory"}</span></header><div class="model-stats"><div><span>Activation matrix</span><strong>${summary.eligible} × ${summary.features}</strong></div><div><span>Median primary N</span><strong>${summary.medianPrimary}</strong></div><div><span>Reconstruction NRMSE</span><strong>${summary.reconstructionNrmse}</strong></div><div><span>Small / unstable</span><strong>${summary.smallOrUnstable}</strong></div></div><p>Saved checkpoints for seeds ${summary.seeds.join(", ")}. M=${summary.m} has a minimum primary-area size of ${summary.minimumPrimary}; small labeled cells remain suppressed here.</p></article>`).join("");
  document.querySelector("#coverage-table tbody").innerHTML = state.data.coverage.yearTrack.map(row => `<tr class="${row.coverageStatus === "partial" ? "partial-coverage-row" : ""}"><td>${row.year}${row.coverageStatus === "partial" ? "*" : ""}<small>${row.coverageStatus === "partial" ? "Partial problem-text coverage" : ""}</small></td><td>${esc(row.track)}</td><td>${row.applications}</td><td>${fmt.format(row.judgeRows)}</td><td>${row.usableProblemText}</td><td>${row.missingProblemText ?? (row.missingSuppressed ? "<10" : "0")}</td></tr>`).join("");
  document.querySelector("#audit-foot").textContent = `Embedding: ${state.data.meta.embeddingModel} · HypotheSAEs commit: ${state.data.meta.upstreamCommit} · ${state.data.meta.privacy}`;
}

function renderAll() {
  ensureSelected();
  renderControls();
  renderRibbon();
  renderFindings();
  renderActionable();
  renderLandscape();
  renderSupportPortfolio();
  renderProfile();
  renderTracks();
  renderFounderLandscape();
  renderJudges();
  renderMethod();
}

function bindControls() {
  document.querySelectorAll("#model-control button").forEach(button => button.addEventListener("click", () => {
    state.model = button.dataset.model; state.selectedId = null; renderAll();
  }));
  document.querySelectorAll("#membership-control button").forEach(button => button.addEventListener("click", () => {
    state.membership = button.dataset.membership; state.selectedId = null; renderAll();
  }));
  document.querySelector("#scope-control").addEventListener("change", event => { state.scope = event.target.value; state.selectedId = null; renderAll(); });
  document.querySelector("#year-control").addEventListener("change", event => { state.year = event.target.value; state.selectedId = null; renderAll(); });
  document.querySelector("#concept-control").addEventListener("change", event => selectConcept(event.target.value));
  document.querySelectorAll("#gender-view-control button").forEach(button => button.addEventListener("click", () => { state.genderView = button.dataset.genderView; renderControls(); renderFounderLandscape(); }));
  document.querySelectorAll("#school-view-control button").forEach(button => button.addEventListener("click", () => { state.schoolView = button.dataset.schoolView; renderControls(); renderFounderLandscape(); }));
}

fetch("data/landscape.json")
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(data => {
    state.data = data;
    state.model = String(data.meta.defaultModel);
    state.membership = data.meta.defaultMembership;
    renderHeader();
    bindControls();
    renderAll();
  })
  .catch(error => {
    document.body.innerHTML = `<main class="shell"><div class="empty-state">The aggregate research payload could not be loaded.<br />${esc(error.message)}</div></main>`;
  });
