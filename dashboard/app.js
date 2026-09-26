// MemVal Dashboard Application Logic

// Register the KaTeX extension so `$...$` (inline) and `$$...$$` (display) math
// in the Markdown docs is rendered by KaTeX. The extension tokenizes math
// before Markdown parsing, so LaTeX (backslashes, subscripts, matrix `\\`) is
// not mangled by marked.
// `nonStandard: true` relaxes the inline `$...$` delimiter rules so math that
// starts a soft-wrapped line or sits next to punctuation (e.g. `($x$)`) still
// renders. Safe here because the docs contain no currency-style `$` text.
if (typeof marked !== 'undefined' && typeof markedKatex !== 'undefined') {
  marked.use(markedKatex({ throwOnError: false, nonStandard: true }));
}

// Convert a Markdown string to sanitized HTML. KaTeX emits MathML plus styled
// spans, so DOMPurify is configured to keep those.
function renderMarkdown(mdText) {
  return DOMPurify.sanitize(marked.parse(mdText), {
    USE_PROFILES: { html: true, mathMl: true, svg: true }
  });
}

const MODELS = [
  { id: 'OriginalEqPropSequenceNetwork', name: 'Original EqProp', icon: 'fa-atom' },
  { id: 'HopfieldSequenceNetwork', name: 'Hopfield Network', icon: 'fa-bolt-lightning', resultsDir: 'AsymmetricHopfieldNetwork' },
  { id: 'GPT2SequenceModel', name: 'GPT-2 Baseline', icon: 'fa-brain' }
];

const BENCHMARKS = [
  // Spatial Suite
  { id: 'tmaze', name: 'T-Maze Completion', suite: 'spatial', icon: 'fa-route', plot: 'tmaze_completion.png', mainMetricKey: 'tmaze_pc_coverage', metricLabel: 'Coverage' },
  { id: 'tmaze_odour', name: 'Odour Disambiguation', suite: 'spatial', icon: 'fa-code-branch', plot: 'tmaze_disambiguation.png', plots: [
      { file: 'tmaze_disambiguation.png', label: 'Fully-cued corner: odour present through the arms' },
      { file: 'tmaze_disambiguation_graded.png', label: 'Graded odour availability vs cue-free delay' }
    ], mainMetricKey: 'tmaze_disamb_full_branch_acc', metricLabel: 'Branch Acc' },
  { id: 'tmaze_reversal', name: 'Reversal', suite: 'spatial', icon: 'fa-rotate-left', plot: 'tmaze_reversal.png', mainMetricKey: 'reversal_extinction_reversal_trials_to_criterion', metricLabel: 'Trials to crit.' },
  // Symbolic Suite
  { id: 'duration', name: 'Convergence Duration', suite: 'symbolic', icon: 'fa-clock', plot: 'convergence_curve.png', mainMetricKey: 'convergence_mrr', metricLabel: 'Conv. MRR' },
  { id: 'length', name: 'Length Sweep', suite: 'symbolic', icon: 'fa-ruler-horizontal', plot: 'length_curves.png', mainMetricKey: 'max_memory_span', metricLabel: 'Max Span' },
  { id: 'multiple_seq', name: 'Multiple Sequences', suite: 'symbolic', icon: 'fa-layer-group', plot: 'multiple_seq_forgetting.png', mainMetricKey: 'delta_mrr_forgetting', metricLabel: 'Delta MRR' },
  { id: 'noise_tolerance', name: 'Noise Tolerance', suite: 'symbolic', icon: 'fa-wave-square', plot: 'noise_invariance.png', mainMetricKey: 'noise_tolerance_threshold', metricLabel: 'Threshold' },
  { id: 'cue_masking', name: 'Cue Masking', suite: 'symbolic', icon: 'fa-mask', plot: 'cue_masking.png', mainMetricKey: 'mask_random_tolerance', metricLabel: 'Mask Tolerance' },
  { id: 'similarity', name: 'Similarity Sweep', suite: 'symbolic', icon: 'fa-object-group', plot: 'semantic_similarity.png', mainMetricKey: 'similarity_effect_mrr_drop', metricLabel: 'MRR Drop' },
  // Symbolic Suite — Proposed (linearity-breaking; not yet implemented). Marked
  // `proposed` so they render as browsable doc pages but stay out of the results
  // matrix and per-model accordions, which depend on real metrics.
  { id: 'high_order_markov', name: 'High-Order Markov', suite: 'symbolic', icon: 'fa-diagram-project', proposed: true, mainMetricKey: 'high_order_recall_acc', metricLabel: 'Order-k Acc' },
  { id: 'context_gating', name: 'Context-Gated (XOR)', suite: 'symbolic', icon: 'fa-shuffle', proposed: true, mainMetricKey: 'context_gating_acc', metricLabel: 'Gating Acc' },
  { id: 'delayed_recall', name: 'Delayed Dependency', suite: 'symbolic', icon: 'fa-backward-step', proposed: true, mainMetricKey: 'delayed_recall_acc', metricLabel: 'n-Back Acc' },
  { id: 'limit_cycle', name: 'Cyclic Free-Running', suite: 'symbolic', icon: 'fa-rotate', proposed: true, mainMetricKey: 'limit_cycle_stable_span', metricLabel: 'Stable Span' }
];

// Benchmarks that have been run and have metrics — used by the results matrix and the
// per-model accordions. Proposed benchmarks are documentation-only until implemented.
const ACTIVE_BENCHMARKS = BENCHMARKS.filter(b => !b.proposed);

// Output directory a model's results/plots are read from. Defaults to the model
// id; a model may override it (e.g. Hopfield reads AsymmetricHopfieldNetwork).
function resultsDirFor(model) {
  return model.resultsDir || model.id;
}

// Normalize a benchmark's plot config to a list of {file, label}. Most benchmarks
// declare a single `plot`; tmaze_odour declares a `plots` array (fully-cued corner
// plus the graded odour-availability sweep).
function getBenchPlots(bench) {
  if (Array.isArray(bench.plots) && bench.plots.length) return bench.plots;
  return [{ file: bench.plot, label: '' }];
}

// Helper to format values
function formatMetricValue(benchmarkId, value) {
  if (value === "TODO" || value === undefined || value === null) return "TODO";
  const num = parseFloat(value);
  if (isNaN(num)) return value;
  
  switch (benchmarkId) {
    case 'tmaze':
    case 'tmaze_odour':
      return (num * 100).toFixed(1) + '%';
    case 'tmaze_reversal':
      return `${num} trials`;
    case 'duration':
    case 'multiple_seq':
    case 'similarity':
      return num.toFixed(2);
    case 'noise_tolerance':
      return num.toFixed(1);
    case 'cue_masking':
      return num.toFixed(2);
    case 'length':
      return `${num} items`;
    default:
      return num.toString();
  }
}

// Helper to get color coding classes
function getMetricClass(benchmarkId, value) {
  if (value === "TODO" || value === undefined || value === null) return '';
  const num = parseFloat(value);
  if (isNaN(num)) return '';

  switch (benchmarkId) {
    case 'tmaze':
      // Coverage: higher is better (non-saturating, unlike the MSE it replaced)
      return num > 0.85 ? 'excellent' : (num > 0.5 ? 'moderate' : 'poor');
    case 'tmaze_odour':
      // Branch accuracy: higher is better
      return num > 0.85 ? 'excellent' : (num > 0.5 ? 'moderate' : 'poor');
    case 'tmaze_reversal':
      // Trials to criterion: fewer is better
      return num <= 2 ? 'excellent' : (num <= 6 ? 'moderate' : 'poor');
    case 'duration':
      return num > 0.85 ? 'excellent' : (num > 0.5 ? 'moderate' : 'poor');
    case 'length':
      return num >= 8 ? 'excellent' : (num >= 5 ? 'moderate' : 'poor');
    case 'multiple_seq':
      return num > -0.15 ? 'excellent' : (num > -0.5 ? 'moderate' : 'poor');
    case 'noise_tolerance':
      return num >= 0.6 ? 'excellent' : (num >= 0.3 ? 'moderate' : 'poor');
    case 'cue_masking':
      // Fraction of the cue discardable with recall still >= 0.5; higher is better.
      return num >= 0.6 ? 'excellent' : (num >= 0.3 ? 'moderate' : 'poor');
    case 'similarity':
      return num < 0.1 ? 'excellent' : (num < 0.4 ? 'moderate' : 'poor');
    default:
      return '';
  }
}

// Global cached state
const state = {
  results: {},
  loading: false,
  error: null
};

// Loader operations
function showLoader() {
  document.getElementById('loader').classList.remove('hidden');
  document.getElementById('app-content').classList.add('hidden');
}

function hideLoader() {
  document.getElementById('loader').classList.add('hidden');
  document.getElementById('app-content').classList.remove('hidden');
}

// Fetch helper functions
async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not fetch ${url}`);
  return await res.text();
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not fetch ${url}`);
  return await res.json();
}

// Aggregates data for all models
async function loadResultsData() {
  if (Object.keys(state.results).length > 0) return; // Already cached
  
  for (const model of MODELS) {
    const dir = resultsDirFor(model);
    state.results[model.id] = {
      spatial: null,
      symbolic: null,
      error: null
    };

    try {
      state.results[model.id].spatial = await fetchJSON(`../results/${dir}/spatial/metrics.json`);
    } catch (err) {
      console.warn(`Failed to load spatial metrics for ${model.id}`, err);
    }

    try {
      state.results[model.id].symbolic = await fetchJSON(`../results/${dir}/symbolic/metrics.json`);
    } catch (err) {
      console.warn(`Failed to load symbolic metrics for ${model.id}`, err);
    }
  }
}

// Update Active Nav Styles
function updateActiveNav(hash) {
  // Remove active from all nav links
  document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
  
  // Find matching nav link
  let activeLink = null;
  if (hash === '' || hash === '#overview') {
    activeLink = document.getElementById('link-overview');
  } else if (hash === '#results') {
    activeLink = document.getElementById('link-results');
  } else if (hash === '#methodology') {
    activeLink = document.getElementById('link-methodology');
  } else if (hash.startsWith('#model/')) {
    const modelId = hash.split('/')[1];
    activeLink = document.getElementById(`link-model-${modelId}`);
  } else if (hash.startsWith('#benchmark/')) {
    const benchId = hash.split('/')[1];
    activeLink = document.getElementById(`link-bench-${benchId}`);
  }
  
  if (activeLink) {
    activeLink.classList.add('active');
  }
}

// Router Logic
async function router() {
  const hash = window.location.hash;
  updateActiveNav(hash);
  showLoader();

  try {
    if (hash === '' || hash === '#overview') {
      await renderOverviewPage();
    } else if (hash === '#results') {
      await renderResultsPage();
    } else if (hash === '#methodology') {
      await renderMethodologyPage();
    } else if (hash.startsWith('#model/')) {
      const modelId = hash.split('/')[1];
      await renderModelPage(modelId);
    } else if (hash.startsWith('#benchmark/')) {
      const benchId = hash.split('/')[1];
      await renderBenchmarkPage(benchId);
    } else {
      // Fallback
      window.location.hash = '#overview';
    }
  } catch (err) {
    document.getElementById('app-content').innerHTML = `
      <div class="card" style="border-color: var(--danger-color); margin-top: 2rem;">
        <h3 style="color: var(--danger-color);"><i class="fa-solid fa-triangle-exclamation"></i> Error loading page</h3>
        <p>${err.message}</p>
        <a href="#overview" class="btn" style="margin-top: 1rem; display: inline-block;">Return Overview</a>
      </div>
    `;
    console.error(err);
  } finally {
    hideLoader();
  }
}

// PAGE RENDERERS

// 1. Overview Page (Project Goals)
async function renderOverviewPage() {
  document.getElementById('page-header-title').innerText = "Overview";
  const contentEl = document.getElementById('app-content');
  
  const mdText = await fetchText('docs/project_goals.md');
  const renderedHTML = renderMarkdown(mdText);
  
  contentEl.innerHTML = `
    <div class="markdown-body">
      ${renderedHTML}
    </div>
  `;
}

// 1b. Evaluation Methodology Page (primary metrics, categories, dimensions)
async function renderMethodologyPage() {
  document.getElementById('page-header-title').innerText = "Evaluation Methodology";
  const contentEl = document.getElementById('app-content');

  const mdText = await fetchText('docs/methodology.md');
  const renderedHTML = renderMarkdown(mdText);

  contentEl.innerHTML = `
    <div class="markdown-body">
      ${renderedHTML}
    </div>
  `;
}

// 2. Results Page (Summary Matrix)
async function renderResultsPage() {
  document.getElementById('page-header-title').innerText = "Results Matrix";
  const contentEl = document.getElementById('app-content');
  
  await loadResultsData();
  
  let tableRows = '';
  for (const model of MODELS) {
    const modelResults = state.results[model.id];
    let cells = '';

    for (const bench of ACTIVE_BENCHMARKS) {
      const suiteResults = modelResults[bench.suite];
      let formatted = "N/A";
      let metricClass = "";
      
      if (suiteResults && suiteResults.metrics) {
        const val = suiteResults.metrics[bench.mainMetricKey];
        formatted = formatMetricValue(bench.id, val);
        metricClass = getMetricClass(bench.id, val);
      }
      
      cells += `
        <td class="text-center">
          <span class="metric-value ${metricClass}">${formatted}</span>
        </td>
      `;
    }
    
    tableRows += `
      <tr>
        <td>
          <a href="#model/${model.id}" class="model-name-link">
            <i class="fa-solid ${model.icon}"></i> ${model.name}
          </a>
        </td>
        ${cells}
      </tr>
    `;
  }
  
  const headers = ACTIVE_BENCHMARKS.map(bench => `
    <th class="text-center">
      <div style="display: flex; flex-direction: column; align-items: center; gap: 0.2rem;">
        <a href="#benchmark/${bench.id}" style="color: inherit; font-size: 0.75rem; hover: text-white">
          <i class="fa-solid ${bench.icon} text-${bench.suite}" style="margin-bottom: 0.2rem;"></i><br/>
          ${bench.name}
        </a>
        <span style="font-size: 0.65rem; color: var(--text-muted); font-weight: normal; text-transform: none;">
          (${bench.metricLabel})
        </span>
      </div>
    </th>
  `).join('');

  contentEl.innerHTML = `
    <div style="margin-bottom: 2rem;">
      <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
        This matrix aggregates performance parameters across spatial and symbolic benchmarking suites. 
        Click on a model name to view detailed architecture and per-benchmark response curves, or click on a benchmark name to understand its cognitive assay and metrics.
      </p>

      <div class="table-responsive">
        <table class="results-table">
          <thead>
            <tr>
              <th>Model</th>
              ${headers}
            </tr>
          </thead>
          <tbody>
            ${tableRows}
          </tbody>
        </table>
      </div>
      
      <!-- Legend -->
      <div class="card" style="margin-top: 1.5rem;">
        <h4 style="font-size: 1rem; margin-bottom: 0.8rem;"><i class="fa-solid fa-circle-question"></i> Performance Threshold Legend</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; font-size: 0.85rem;">
          <div>
            <span class="status-dot" style="background-color: #52c41a; box-shadow: 0 0 4px #52c41a;"></span>
            <strong style="color: #52c41a;">Excellent</strong>: Highly stable, low reconstruction error
          </div>
          <div>
            <span class="status-dot" style="background-color: #faad14; box-shadow: 0 0 4px #faad14;"></span>
            <strong style="color: #faad14;">Moderate</strong>: Partial recall capacity or minor decay
          </div>
          <div>
            <span class="status-dot" style="background-color: #ff4d4f; box-shadow: 0 0 4px #ff4d4f;"></span>
            <strong style="color: #ff4d4f;">Poor</strong>: Complete recall failure / interference
          </div>
        </div>
      </div>
    </div>
  `;
}

// 3. Model Detail Page
async function renderModelPage(modelId) {
  const modelInfo = MODELS.find(m => m.id === modelId);
  if (!modelInfo) throw new Error(`Model ${modelId} not found`);
  
  document.getElementById('page-header-title').innerText = `${modelInfo.name} Details`;
  const contentEl = document.getElementById('app-content');
  
  // Load Markdown
  const mdText = await fetchText(`docs/models/${modelId}.md`);
  const renderedHTML = renderMarkdown(mdText);
  
  // Load Results for Collapsible section
  await loadResultsData();
  const results = state.results[modelId];
  
  // Build Accordion items
  let accordionItemsHTML = '';

  for (const bench of ACTIVE_BENCHMARKS) {
    const suiteData = results[bench.suite];
    
    if (!suiteData || !suiteData.metrics) {
      accordionItemsHTML += `
        <div class="accordion-item" id="accordion-${bench.id}">
          <div class="accordion-header" onclick="toggleAccordion('${bench.id}')">
            <div class="accordion-header-left">
              <i class="fa-solid ${bench.icon} text-${bench.suite}"></i>
              <span>${bench.name}</span>
              <span class="badge badge-${bench.suite}" style="margin-left: 0.5rem;">${bench.suite}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
              <span class="metric-value" style="color: var(--text-muted);">N/A</span>
              <i class="fa-solid fa-chevron-down chevron-icon"></i>
            </div>
          </div>
          <div class="accordion-content">
            <div class="card" style="border-color: rgba(255, 255, 255, 0.05); text-align: center; padding: 2rem;">
              <i class="fa-solid fa-circle-minus" style="font-size: 2rem; color: var(--text-muted); margin-bottom: 1rem;"></i>
              <p style="color: var(--text-muted);">This benchmark suite (${bench.suite}) has not been executed for this model.</p>
            </div>
          </div>
        </div>
      `;
      continue;
    }
    
    const metricsObj = suiteData.metrics;
    
    // Aggregate specific metrics to display in this benchmark details
    let metricBreakdownHTML = '';
    const mainVal = metricsObj[bench.mainMetricKey];
    const formattedMainVal = formatMetricValue(bench.id, mainVal);
    const mainMetricClass = getMetricClass(bench.id, mainVal);
    
    if (bench.suite === 'spatial') {
      if (bench.id === 'tmaze') {
        const cov = formatMetricValue('tmaze', metricsObj['tmaze_pc_coverage']);
        const mse = metricsObj['tmaze_pc_mse'];
        const mseStr = (typeof mse === 'number') ? mse.toFixed(4) : 'N/A';
        const divStep = metricsObj['tmaze_pc_divergence_step'];
        const recallLen = metricsObj['tmaze_pc_recall_len'];
        const divStr = (divStep === null || divStep === undefined)
          ? 'N/A' : `${divStep} / ${recallLen}`;
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Coverage<br/><small style="opacity:0.7">(non-saturating; keeps its range when recall collapses)</small></div>
              <div class="detail-metric-value ${getMetricClass('tmaze', metricsObj['tmaze_pc_coverage'])}">${cov}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Divergence step<br/><small style="opacity:0.7">(first step off the path; span analogue)</small></div>
              <div class="detail-metric-value">${divStr}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Completion MSE<br/><small style="opacity:0.7">(saturates in failure — read beside coverage)</small></div>
              <div class="detail-metric-value">${mseStr}</div>
            </div>
          </div>
        `;
      } else if (bench.id === 'tmaze_odour') {
        const fullAcc  = formatMetricValue('tmaze_odour', metricsObj['tmaze_disamb_full_branch_acc']);
        const mecAcc   = formatMetricValue('tmaze_odour', metricsObj['tmaze_disamb_mec_only_branch_acc']);
        const fullConf = formatMetricValue('tmaze_odour', metricsObj['tmaze_disamb_full_confusion']);
        const fullAccClass = getMetricClass('tmaze_odour', metricsObj['tmaze_disamb_full_branch_acc']);
        const mecAccClass  = getMetricClass('tmaze_odour', metricsObj['tmaze_disamb_mec_only_branch_acc']);
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Full recall branch accuracy<br/><small style="opacity:0.7">(place cells + odour cue)</small></div>
              <div class="detail-metric-value ${fullAccClass}">${fullAcc}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">MEC-only branch accuracy<br/><small style="opacity:0.7">(place cells only, odour masked)</small></div>
              <div class="detail-metric-value ${mecAccClass}">${mecAcc}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Full recall confusion rate</div>
              <div class="detail-metric-value">${fullConf}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Concurrent divergence accuracy<br/><small style="opacity:0.7">(odour available at the branch — the fully-cued reference)</small></div>
              <div class="detail-metric-value ${getMetricClass('tmaze_odour', metricsObj['tmaze_disamb_graded_concurrent_divergence_accuracy'])}">${formatMetricValue('tmaze_odour', metricsObj['tmaze_disamb_graded_concurrent_divergence_accuracy'])}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Withdrawn, delay 0<br/><small style="opacity:0.7">(odour ends at the last stem step)</small></div>
              <div class="detail-metric-value ${getMetricClass('tmaze_odour', metricsObj['tmaze_disamb_graded_delay0_divergence_accuracy'])}">${formatMetricValue('tmaze_odour', metricsObj['tmaze_disamb_graded_delay0_divergence_accuracy'])}</div>
            </div>
          </div>
          <p style="font-size:0.85rem; opacity:0.8; margin-top:0.6rem;">
            Withdrawn rows are expected at chance under the memoryless
            <code>predict_next</code> probe: no current arm carries state across the
            cue-free stretch, so the floor is a <em>protocol</em> finding, not a model ranking.
          </p>
        `;
      } else if (bench.id === 'tmaze_reversal') {
        const ttc = metricsObj['reversal_extinction_reversal_trials_to_criterion'];
        const ttcStr = (ttc === null || ttc === undefined) ? 'N/A' : `${ttc} trials`;
        const persev = metricsObj['reversal_extinction_reversal_final_perseveration'];
        const persevStr = (typeof persev === 'number') ? (persev * 100).toFixed(1) + '%' : 'N/A';
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Trials to criterion (reversal stage)</div>
              <div class="detail-metric-value ${getMetricClass('tmaze_reversal', ttc)}">${ttcStr}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Final perseveration<br/><small style="opacity:0.7">(the failure this section detects: over-preservation, not forgetting)</small></div>
              <div class="detail-metric-value">${persevStr}</div>
            </div>
          </div>
        `;
      }
      // Symbolic breakdowns
      if (bench.id === 'duration') {
        const mrrVal = metricsObj['convergence_mrr'];
        const mrr = formatMetricValue(bench.id, mrrVal);
        const span = formatMetricValue('length', metricsObj['convergence_span']);
        const epochs = metricsObj['convergence_epochs'];
        const converged = metricsObj['converged'];

        let epochsVal = epochs !== null && epochs !== undefined ? `${epochs} ep` : 'Did not converge';
        let statusClass = converged ? 'excellent' : 'poor';
        let statusText = converged ? '✓ converged' : '⚠ Threshold 0.95 not reached';

        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">${converged ? 'Convergence MRR' : 'Best MRR'}</div>
              <div class="detail-metric-value ${getMetricClass('duration', mrrVal)}">${mrr}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Convergence Epochs</div>
              <div class="detail-metric-value ${statusClass}">${epochsVal}</div>
              <div style="font-size: 0.8rem; margin-top: 0.25rem; opacity: 0.85;">${statusText}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Memory Span</div>
              <div class="detail-metric-value">${span}</div>
            </div>
          </div>
        `;
      } else if (bench.id === 'length') {
        const span = formatMetricValue(bench.id, metricsObj['max_memory_span']);
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Max Memory Span</div>
              <div class="detail-metric-value ${getMetricClass('length', metricsObj['max_memory_span'])}">${span}</div>
            </div>
          </div>
        `;
      } else if (bench.id === 'multiple_seq') {
        const delta = formatMetricValue(bench.id, metricsObj['delta_mrr_forgetting']);
        const mrrBefore = formatMetricValue(bench.id, metricsObj['multiple_seq_mrr_before']);
        const mrrAfter = formatMetricValue(bench.id, metricsObj['multiple_seq_mrr_after']);
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Forgetting Delta MRR</div>
              <div class="detail-metric-value ${getMetricClass('multiple_seq', metricsObj['delta_mrr_forgetting'])}">${delta}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">MRR (Before consolidation)</div>
              <div class="detail-metric-value">${mrrBefore}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">MRR (After consolidation)</div>
              <div class="detail-metric-value">${mrrAfter}</div>
            </div>
          </div>
        `;
      } else if (bench.id === 'noise_tolerance') {
        const threshold = formatMetricValue(bench.id, metricsObj['noise_tolerance_threshold']);
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Noise Tolerance Threshold</div>
              <div class="detail-metric-value ${getMetricClass('noise_tolerance', metricsObj['noise_tolerance_threshold'])}">${threshold}</div>
            </div>
          </div>
        `;
      } else if (bench.id === 'cue_masking') {
        const tol = formatMetricValue(bench.id, metricsObj['mask_random_tolerance']);
        const inb = formatMetricValue(bench.id, metricsObj['mask_random_recall_in_bound']);
        const asym = formatMetricValue(bench.id, metricsObj['mask_block_asymmetry']);
        const adv = formatMetricValue(bench.id, metricsObj['mask_random_completion_advantage']);
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Mask Tolerance</div>
              <div class="detail-metric-value ${getMetricClass('cue_masking', metricsObj['mask_random_tolerance'])}">${tol}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">In-bound Recall</div>
              <div class="detail-metric-value ${getMetricClass('cue_masking', metricsObj['mask_random_recall_in_bound'])}">${inb}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Completion Advantage</div>
              <div class="detail-metric-value">${adv}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">Block Asymmetry</div>
              <div class="detail-metric-value">${asym}</div>
            </div>
          </div>
        `;
      } else if (bench.id === 'similarity') {
        const drop = formatMetricValue(bench.id, metricsObj['similarity_effect_mrr_drop']);
        const high = formatMetricValue(bench.id, metricsObj['mrr_high_similarity']);
        const low = formatMetricValue(bench.id, metricsObj['mrr_low_similarity']);
        metricBreakdownHTML = `
          <div class="metrics-detail-grid">
            <div class="detail-metric-card">
              <div class="detail-metric-label">Similarity MRR Drop</div>
              <div class="detail-metric-value ${getMetricClass('similarity', metricsObj['similarity_effect_mrr_drop'])}">${drop}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">MRR (High Overlap)</div>
              <div class="detail-metric-value">${high}</div>
            </div>
            <div class="detail-metric-card">
              <div class="detail-metric-label">MRR (Low Overlap)</div>
              <div class="detail-metric-value">${low}</div>
            </div>
          </div>
        `;
      }
    }

    const plotsHTML = getBenchPlots(bench).map(p => {
      const plotPath = `../results/${resultsDirFor(modelInfo)}/${bench.suite}/plots/${p.file}`;
      const caption = p.label
        ? `Figure: ${p.label} — ${modelInfo.name}, ${bench.name}.`
        : `Figure: Response curves generated by ${modelInfo.name} during the ${bench.name} assay.`;
      return `
          <div class="plot-container">
            <img src="${plotPath}" class="plot-image" alt="${bench.name} — ${p.label || 'evaluation curves'}" onerror="this.style.display='none'; this.nextElementSibling.innerText='Plot not available for this run.'">
            <div class="plot-caption">${caption}</div>
          </div>`;
    }).join('');

    accordionItemsHTML += `
      <div class="accordion-item" id="accordion-${bench.id}">
        <div class="accordion-header" onclick="toggleAccordion('${bench.id}')">
          <div class="accordion-header-left">
            <i class="fa-solid ${bench.icon} text-${bench.suite}"></i>
            <span>${bench.name}</span>
            <span class="badge badge-${bench.suite}" style="margin-left: 0.5rem;">${bench.suite}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 1rem;">
            <span class="metric-value ${mainMetricClass}">${bench.metricLabel}: ${formattedMainVal}</span>
            <i class="fa-solid fa-chevron-down chevron-icon"></i>
          </div>
        </div>
        <div class="accordion-content">
          ${metricBreakdownHTML}
          ${plotsHTML}
        </div>
      </div>
    `;
  }
  
  contentEl.innerHTML = `
    <div class="markdown-body">
      ${renderedHTML}
    </div>
    
    <hr/>
    
    <div class="accordion-section">
      <h2 style="font-family: 'Outfit', sans-serif; margin-bottom: 1.5rem;">Benchmark Performance Details</h2>
      <p style="color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.9rem;">
        Click on any benchmark section below to expand detailed scoring metrics and view response curves (loss curves, spatial trajectories, or forgetting rates) generated dynamically for this model.
      </p>
      <div class="accordion">
        ${accordionItemsHTML}
      </div>
    </div>
  `;
}

// 4. Benchmark Detail Page
async function renderBenchmarkPage(benchId) {
  const benchInfo = BENCHMARKS.find(b => b.id === benchId);
  if (!benchInfo) throw new Error(`Benchmark ${benchId} not found`);
  
  document.getElementById('page-header-title').innerText = `${benchInfo.name} Benchmark`;
  const contentEl = document.getElementById('app-content');
  
  // Load Markdown
  const mdText = await fetchText(`docs/benchmarks/${benchId}.md`);
  const renderedHTML = renderMarkdown(mdText);

  // Proposed benchmarks are documentation-only: render the assay description plus a
  // status note, and skip the cross-model comparison (there are no results to show).
  if (benchInfo.proposed) {
    contentEl.innerHTML = `
      <div class="markdown-body">
        ${renderedHTML}
      </div>

      <hr/>

      <div class="card" style="margin-top: 2rem; border-color: var(--border-color);">
        <h3 style="margin-top: 0;"><i class="fa-solid fa-flask"></i> Proposed benchmark</h3>
        <p style="color: var(--text-muted); margin-bottom: 0;">
          This is a proposed <em>linearity-breaking</em> assay — see the
          <a href="#methodology">Evaluation Methodology</a> page for how it fits the
          symbolic suite. It has not been implemented yet, so no cross-model results are
          available.
        </p>
      </div>
    `;
    return;
  }

  // Find performance of all models on this benchmark to show a comparative card
  await loadResultsData();

  let comparisonCards = '';
  for (const model of MODELS) {
    const suiteData = state.results[model.id]?.[benchInfo.suite];
    
    if (!suiteData || !suiteData.metrics) {
      comparisonCards += `
        <div class="card" style="margin-top: 1rem; opacity: 0.65;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; font-size: 1.1rem; color: var(--text-muted);"><i class="fa-solid ${model.icon}"></i> ${model.name}</h4>
            <span class="metric-value" style="font-size: 1.2rem; color: var(--text-muted);">N/A</span>
          </div>
          <div class="card" style="border-color: rgba(255, 255, 255, 0.05); text-align: center; padding: 1.5rem; background: rgba(0,0,0,0.15);">
            <p style="color: var(--text-muted); font-size: 0.85rem;">No results available for this benchmark suite.</p>
          </div>
        </div>
      `;
      continue;
    }

    const val = suiteData.metrics[benchInfo.mainMetricKey];
    const formatted = formatMetricValue(benchInfo.id, val);
    const mClass = getMetricClass(benchInfo.id, val);
    const plotsHTML = getBenchPlots(benchInfo).map(p => {
      const plotPath = `../results/${resultsDirFor(model)}/${benchInfo.suite}/plots/${p.file}`;
      return `
        <div class="plot-container">
          <img src="${plotPath}" class="plot-image" style="max-height: 250px;" alt="${model.name} on ${benchInfo.name}${p.label ? ' — ' + p.label : ''}" onerror="this.style.display='none'; this.nextElementSibling.innerText='Plot not generated.'">
          <div class="plot-caption">${p.label ? p.label + ' — ' : ''}Evaluation plot for ${model.name}</div>
        </div>`;
    }).join('');

    comparisonCards += `
      <div class="card" style="margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
          <h4 style="margin: 0; font-size: 1.1rem;"><i class="fa-solid ${model.icon}"></i> ${model.name}</h4>
          <span class="metric-value ${mClass}" style="font-size: 1.2rem;">${benchInfo.metricLabel}: ${formatted}</span>
        </div>
        ${plotsHTML}
      </div>
    `;
  }
  
  contentEl.innerHTML = `
    <div class="markdown-body">
      ${renderedHTML}
    </div>
    
    <hr/>
    
    <div style="margin-top: 2rem;">
      <h2 style="font-family: 'Outfit', sans-serif;">Cross-Model Comparisons</h2>
      <p style="color: var(--text-muted); font-size: 0.9rem;">
        Comparison of all candidate models on the ${benchInfo.name} task, including quantitative outputs and output plots.
      </p>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
        ${comparisonCards}
      </div>
    </div>
  `;
}

// Global toggle helper for accordions (attached to window so onclick works)
window.toggleAccordion = function(benchId) {
  const item = document.getElementById(`accordion-${benchId}`);
  if (item) {
    item.classList.toggle('open');
  }
};

// Event Listeners
window.addEventListener('hashchange', router);
window.addEventListener('DOMContentLoaded', router);
