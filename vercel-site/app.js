const state = {
  data: null,
  selectedId: null,
  genderMode: "representation",
};

const TRACKS = ["Open", "Social Impact", "Health & Life Sciences"];
const TRACK_COLORS = {
  "Open": [49, 121, 132],
  "Social Impact": [189, 139, 67],
  "Health & Life Sciences": [165, 28, 48],
};
const fmt = new Intl.NumberFormat("en-US");
const pct = value => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const number = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const signed = (value, digits = 2) => value == null ? "—" : `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
const esc = value => String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function meeting() { return state.data.meeting; }
function problems() { return meeting().problems; }
function selectedProblem() { return problems().find(item => item.id === state.selectedId) || problems()[0]; }

function selectProblem(id, destination = null) {
  const next = problems().find(item => item.id === Number(id));
  if (!next) return;
  state.selectedId = next.id;
  const url = new URL(window.location.href);
  url.searchParams.set("problem", String(next.id));
  history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  renderSharedProblems();
  renderExplore();
  renderSupport();
  if (destination) document.querySelector(destination).scrollIntoView({ behavior: "smooth", block: "start" });
}

function heatStyle(track, share) {
  const [r, g, b] = TRACK_COLORS[track];
  const alpha = clamp(.08 + (Number(share) / .4) * .48, .08, .56);
  return `background-color:rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

function trackCell(track, cell) {
  if (!cell || cell.suppressed) {
    return `<td class="heat-cell suppressed" aria-label="${esc(track)} cell suppressed"><strong>Suppressed</strong><small>N &lt; 10 · not zero</small></td>`;
  }
  return `<td class="heat-cell" style="${heatStyle(track, cell.share)}"><strong>${pct(cell.share)}</strong><small>N=${fmt.format(cell.n)} applications</small></td>`;
}

function renderSharedProblems() {
  const denominators = meeting().trackDenominators;
  document.querySelector("#open-denominator").textContent = `n=${fmt.format(denominators.Open)} usable applications`;
  document.querySelector("#social-denominator").textContent = `n=${fmt.format(denominators["Social Impact"])} usable applications`;
  document.querySelector("#health-denominator").textContent = `n=${fmt.format(denominators["Health & Life Sciences"])} usable applications`;
  const rows = problems().map((problem, index) => `
    <tr data-id="${problem.id}" tabindex="0" role="button" aria-pressed="${problem.id === state.selectedId}" class="${problem.id === state.selectedId ? "selected" : ""}">
      <td class="problem-cell"><span>${String(index + 1).padStart(2, "0")}</span><strong>${esc(problem.label)}</strong></td>
      <td class="portfolio-cell"><strong>${fmt.format(problem.n)}</strong><small>${pct(problem.share)} of portfolio</small></td>
      ${TRACKS.map(track => trackCell(track, problem.tracks[track])).join("")}
    </tr>`).join("");
  document.querySelector("#shared-problem-rows").innerHTML = rows;
  document.querySelectorAll("#shared-problem-rows tr[data-id]").forEach(row => {
    row.addEventListener("click", () => selectProblem(row.dataset.id));
    row.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectProblem(row.dataset.id);
      }
    });
  });
  const current = selectedProblem();
  document.querySelector("#explore-selected").innerHTML = `Explore ${esc(current.label)} <span aria-hidden="true">→</span>`;
}

function barRow({ label, detail, share, n, suppressed = false, partial = false }, maxShare, unit = "applications") {
  const width = suppressed ? 0 : clamp((Number(share) / Math.max(maxShare, .001)) * 100, 2, 100);
  return `<div class="bar-row ${suppressed ? "suppressed" : ""} ${partial ? "partial" : ""}">
    <div class="bar-label">${esc(label)}${detail ? `<small>${esc(detail)}</small>` : ""}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
    <div class="bar-value">${suppressed ? "Suppressed" : `${pct(share)} · N=${fmt.format(n)} ${esc(unit)}`}</div>
  </div>`;
}

function renderYears(problem) {
  const visible = problem.years.filter(item => !item.suppressed);
  const maxShare = Math.max(.01, ...visible.map(item => item.share));
  document.querySelector("#year-lens").innerHTML = problem.years.map(item => barRow({
    label: String(item.year),
    detail: item.year === 2021 ? "Partial problem-text coverage" : "Usable applications in year",
    share: item.share,
    n: item.n,
    suppressed: item.suppressed,
    partial: item.year === 2021,
  }, maxShare)).join("");
}

function renderTracks(problem) {
  const visible = TRACKS.map(track => problem.tracks[track]).filter(item => item && !item.suppressed);
  const maxShare = Math.max(.01, ...visible.map(item => item.share));
  document.querySelector("#track-lens").innerHTML = TRACKS.map(track => {
    const cell = problem.tracks[track];
    return barRow({
      label: track,
      detail: `${fmt.format(meeting().trackDenominators[track])} usable track applications`,
      share: cell?.share,
      n: cell?.n,
      suppressed: !cell || cell.suppressed,
    }, maxShare);
  }).join("");
}

function renderSchools(problem) {
  if (!problem.schools.length) {
    document.querySelector("#school-lens").innerHTML = `<div class="empty-message">No school-affiliation cell meets the public threshold for this problem.</div>`;
    return;
  }
  const maxN = Math.max(...problem.schools.map(item => item.n));
  document.querySelector("#school-lens").innerHTML = problem.schools.slice(0, 6).map(item => barRow({
    label: item.label,
    detail: "Reported affiliation",
    share: item.n / maxN,
    n: item.n,
  }, 1, "founder records").replace(`${pct(item.n / maxN)} · `, "")).join("");
}

function renderGenderRepresentation(problem) {
  const gender = problem.leadGender;
  const groups = gender.groups.map(group => `<div class="gender-group">
    <span>${esc(group.label)}</span>
    <strong>${fmt.format(group.n)}</strong>
    <small>${pct(group.shareAmongReported)} of reported lead-applicant gender</small>
    <div class="gender-share"><i style="width:${clamp(group.shareAmongReported * 100, 0, 100)}%"></i></div>
  </div>`).join("");
  document.querySelector("#gender-lens").innerHTML = `<div class="gender-summary">
    <p class="gender-coverage">Reporting denominator: ${fmt.format(gender.reportedN)} of ${fmt.format(gender.problemN)} applications (${pct(gender.reportingCoverage)}). Supplied field only; gender is never inferred.</p>
    <div class="gender-groups">${groups || `<div class="empty-message">Published female/male cells are unavailable.</div>`}</div>
    <p class="lens-note">Shares use all reported lead-gender responses as the denominator. Other response categories may be suppressed, so displayed shares need not sum to 100%.</p>
  </div>`;
}

function evaluationGroup(group) {
  return `<article class="evaluation-group">
    <h4>${esc(group.label)} · N=${fmt.format(group.n)}</h4>
    <dl>
      <div><dt>Raw Recommendation / 5</dt><dd>${number(group.rawMean)}<br /><small>95% CI ${number(group.rawCiLow)}–${number(group.rawCiHigh)}</small></dd></div>
      <div><dt>Year×track adjusted</dt><dd>${signed(group.adjustedMean)}<br /><small>95% CI ${signed(group.adjustedCiLow)} to ${signed(group.adjustedCiHigh)}</small></dd></div>
    </dl>
  </article>`;
}

function renderGenderEvaluation(problem) {
  const evaluation = problem.leadGender.evaluation;
  if (!evaluation.available) {
    document.querySelector("#gender-lens").innerHTML = `<div class="gender-summary"><div class="empty-message">${esc(evaluation.message)}</div><p class="evaluation-caveat">Descriptive only; differences may reflect applicant, judge, cohort, or other composition.</p></div>`;
    return;
  }
  document.querySelector("#gender-lens").innerHTML = `<div class="gender-summary">
    <div class="evaluation-groups">${evaluation.groups.map(evaluationGroup).join("")}</div>
    <div class="difference-callout"><span>${esc(evaluation.differenceLabel)}</span><strong>Female − male adjusted: ${signed(evaluation.adjustedDifferenceFemaleMinusMale)}</strong><small>Bootstrap 95% CI ${signed(evaluation.adjustedDifferenceCiLow)} to ${signed(evaluation.adjustedDifferenceCiHigh)}</small></div>
    <p class="evaluation-caveat"><b>Descriptive, not causal.</b> Differences may reflect applicant, judge, cohort, selection, or other composition. Recommendation is a judging score, not a funding outcome.</p>
  </div>`;
}

function renderGender(problem) {
  document.querySelectorAll("[data-gender-mode]").forEach(button => button.classList.toggle("active", button.dataset.genderMode === state.genderMode));
  if (state.genderMode === "evaluation") renderGenderEvaluation(problem);
  else renderGenderRepresentation(problem);
}

function renderExplore() {
  const problem = selectedProblem();
  const select = document.querySelector("#problem-select");
  select.innerHTML = problems().map(item => `<option value="${item.id}">${esc(item.label)}</option>`).join("");
  select.value = String(problem.id);
  document.querySelector("#problem-title").textContent = problem.label;
  document.querySelector("#problem-description").textContent = problem.description;
  document.querySelector("#problem-n").textContent = fmt.format(problem.n);
  document.querySelector("#problem-share").textContent = pct(problem.share);
  renderYears(problem);
  renderTracks(problem);
  renderSchools(problem);
  renderGender(problem);
}

function scorePosition(score) {
  return clamp(((Number(score) - 2.5) / 1.7) * 100, 2, 98);
}

function renderSupport() {
  document.querySelector("#support-rows").innerHTML = problems().map(problem => {
    const problemPosition = scorePosition(problem.support.problem);
    const businessPosition = scorePosition(problem.support.business);
    const left = Math.min(problemPosition, businessPosition);
    const width = Math.abs(problemPosition - businessPosition);
    return `<button type="button" class="support-row ${problem.id === state.selectedId ? "selected" : ""}" data-id="${problem.id}">
      <span class="support-row-label"><strong>${esc(problem.label)}</strong><small>N=${fmt.format(problem.support.n)} applications</small></span>
      <span class="paired-scale" aria-label="Problem score ${number(problem.support.problem)}, Business Model score ${number(problem.support.business)}">
        <i class="paired-rail"></i><i class="paired-span" style="left:${left}%;width:${width}%"></i>
        <i class="score-marker problem" style="left:${problemPosition}%" title="Problem & Customer ${number(problem.support.problem)}"></i>
        <i class="score-marker business" style="left:${businessPosition}%" title="Business Model ${number(problem.support.business)}"></i>
      </span>
      <span class="support-gap-value"><span>Gap</span><strong>+${number(problem.support.gap)}</strong></span>
    </button>`;
  }).join("");
  document.querySelectorAll("#support-rows [data-id]").forEach(button => button.addEventListener("click", () => selectProblem(button.dataset.id)));

  const problem = selectedProblem();
  const support = problem.support;
  document.querySelector("#support-problem-title").textContent = problem.label;
  document.querySelector("#support-problem-score").textContent = `${number(support.problem)} / 5`;
  document.querySelector("#support-business-score").textContent = `${number(support.business)} / 5`;
  document.querySelector("#support-gap").textContent = `+${number(support.gap)}`;
  document.querySelector("#support-n").textContent = `Application-level means · N=${fmt.format(support.n)} applications`;
  document.querySelector("#support-theme").textContent = support.theme;
  document.querySelector("#support-question").textContent = support.question;
  document.querySelector("#support-validation").textContent = support.validation;
}

function bindControls() {
  document.querySelector("#explore-selected").addEventListener("click", () => document.querySelector("#explore").scrollIntoView({ behavior: "smooth" }));
  document.querySelector("#support-selected").addEventListener("click", () => document.querySelector("#support").scrollIntoView({ behavior: "smooth" }));
  document.querySelector("#problem-select").addEventListener("change", event => selectProblem(event.target.value));
  document.querySelectorAll("[data-gender-mode]").forEach(button => button.addEventListener("click", () => {
    state.genderMode = button.dataset.genderMode;
    renderGender(selectedProblem());
  }));
}

function renderAll() {
  document.querySelector("#meta-applications").textContent = fmt.format(state.data.meta.applications);
  document.querySelector("#meta-eligible").textContent = fmt.format(state.data.meta.eligibleProblemTexts);
  renderSharedProblems();
  renderExplore();
  renderSupport();
}

fetch("data/landscape.json")
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(data => {
    state.data = data;
    const requested = Number(new URL(window.location.href).searchParams.get("problem"));
    const valid = data.meeting.problems.some(item => item.id === requested);
    state.selectedId = valid ? requested : data.meeting.defaultProblemId;
    bindControls();
    renderAll();
  })
  .catch(error => {
    document.body.innerHTML = `<main class="shell"><div class="empty-message" style="margin-top:60px">The aggregate meeting payload could not be loaded.<br />${esc(error.message)}</div></main>`;
  });
