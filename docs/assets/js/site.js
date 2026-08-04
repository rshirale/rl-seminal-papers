// ── Decorative SVGs: mark as hidden from assistive tech ──────────────────
document.querySelectorAll('a svg, button svg, .hero-badge svg, .cta-chip svg, .learn-icon svg, .code-panel-header svg, .footer-brand svg').forEach(svg => {
  if (!svg.getAttribute('aria-label')) svg.setAttribute('aria-hidden', 'true');
});

// ── Reduced-motion preference ────────────────────────────────────────────
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── Analytics consent ────────────────────────────────────────────────────
const consentBanner = document.getElementById('consentBanner');
const consentKey = 'rl-seminal-papers-analytics-consent';
function trackEvent(name, parameters = {}) {
  if (analyticsEnabled) gtag('event', name, parameters);
}
let savedConsent = null;
try { savedConsent = localStorage.getItem(consentKey); } catch (_) {}
if (savedConsent === 'accepted') loadAnalytics();
if (!savedConsent) consentBanner.hidden = false;
document.getElementById('acceptAnalytics').addEventListener('click', () => {
  try { localStorage.setItem(consentKey, 'accepted'); } catch (_) {}
  consentBanner.hidden = true; loadAnalytics();
});
document.getElementById('rejectAnalytics').addEventListener('click', () => {
  try { localStorage.setItem(consentKey, 'declined'); } catch (_) {}
  consentBanner.hidden = true;
});

// ── Theme preference ─────────────────────────────────────────────────────
const themeButtons = [document.getElementById('themeToggle'), document.getElementById('mobileThemeToggle')];
function updateThemeControls() {
  const light = document.documentElement.dataset.theme === 'light';
  themeButtons.forEach(button => {
    button.textContent = light ? '☾ Dark' : '☼ Light';
    button.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
  });
  document.querySelector('meta[name="theme-color"]').setAttribute('content', light ? '#fffaf5' : '#141110');
}
themeButtons.forEach(button => button.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('rl-seminal-papers-theme', next); } catch (_) {}
  updateThemeControls();
  trackEvent('theme_changed', { theme: next });
}));
updateThemeControls();

// ── Q-Value Heatmap ──────────────────────────────────────────────────────
const qGrid = [
  [0.05,0.15,0.38,0.60,0.82,1.00],
  [0.04,0.18,0.40,0.62,0.78,0.84],
  [0.02,0.12,0.32,0.54,0.68,0.74],
  [0.01,0.07,0.22,0.42,0.58,0.64],
  [0.00,0.04,0.14,0.32,0.48,0.52],
  [0.00,0.01,0.06,0.18,0.30,0.38],
];
const agentPath = [
  [5,0],[4,0],[3,0],[2,0],[1,0],[0,0],
  [0,1],[0,2],[0,3],[0,4],[0,5]
];
const gridEl = document.getElementById('qGrid');
gridEl.innerHTML = ''; // Clear existing content
const cells = [];

for (let r = 0; r < 6; r++) {
  for (let c = 0; c < 6; c++) {
    const isGoal = r === 0 && c === 5;
    const val = qGrid[r][c];
    const alpha = val * 0.72 + 0.07;
    const div = document.createElement('div');
    div.className = 'q-cell';
    div.style.backgroundColor = isGoal ? 'var(--primary)' : `rgba(255,183,77,${alpha})`;
    div.title = isGoal ? 'Goal, value 1.00' : `Q-value ${val.toFixed(2)}`;
    if (isGoal) div.innerHTML = '<span class="q-star" aria-hidden="true">★</span>';
    gridEl.appendChild(div);
    cells.push({ el: div, isGoal, r, c });
  }
}

let step = 0;
let userPaused = reducedMotion;
let autoPaused = false;
let timer = null;

function updateGrid() {
  const [ar, ac] = agentPath[step];
  cells.forEach(({ el, isGoal, r, c }) => {
    const isAgent = r === ar && c === ac;
    el.classList.toggle('is-agent', isAgent && !isGoal);
    el.classList.toggle('is-goal', isGoal);
    if (!isGoal) el.innerHTML = isAgent ? '<div class="q-dot" aria-hidden="true"></div>' : '';
  });
}

function tick() {
  if (userPaused || autoPaused) return;
  step++;
  if (step >= agentPath.length) {
    step = agentPath.length - 1;
    autoPaused = true;
    timer = setTimeout(() => {
      step = 0;
      autoPaused = false;
      updateGrid();
      scheduleNext();
    }, 1400);
    updateGrid();
    return;
  }
  updateGrid();
  scheduleNext();
}

function scheduleNext() {
  if (!userPaused && !autoPaused) timer = setTimeout(tick, 520);
}

const pauseBtn = document.getElementById('qPauseBtn');
pauseBtn.addEventListener('click', () => {
  userPaused = !userPaused;
  pauseBtn.textContent = userPaused ? 'Play' : 'Pause';
  pauseBtn.setAttribute('aria-label', userPaused ? 'Play Q-value animation' : 'Pause Q-value animation');
  pauseBtn.setAttribute('aria-pressed', userPaused);
  if (!userPaused) { 
    autoPaused = false; 
    scheduleNext(); 
  } else { 
    clearTimeout(timer); 
  }
});

updateGrid();
if (reducedMotion) {
  pauseBtn.textContent = 'Play';
  pauseBtn.setAttribute('aria-label', 'Play Q-value animation');
  pauseBtn.setAttribute('aria-pressed', 'true');
} else {
  scheduleNext();
}

// ── Paper-to-code switcher ──────────────────────────────────────────────
const paperExamples = {
  td0: {
    title: 'Sutton (1988) — TD(0) Update Rule', codeTitle: 'Python · Chapter 2',
    formula: 'V(S<sub>t</sub>) ← V(S<sub>t</sub>) + α [R<sub>t+1</sub> + γ·V(S<sub>t+1</sub>) − V(S<sub>t</sub>)]',
    note: 'Temporal-difference error drives value updates — no model, no Monte Carlo rollout.',
    code: '<span class="c-comment"># Bootstrapped TD error</span>\n<span class="c-var">td_error = (</span>\n<span class="c-var">    reward + gamma * V[next_s]</span>\n<span class="c-var">) - V[s]</span>\n\n<span class="c-accent">V[s]</span><span class="c-var"> += alpha * td_error</span>'
  },
  qlearning: {
    title: 'Watkins (1989) — Q-Learning Update', codeTitle: 'Python · Chapter 2',
    formula: 'Q(S<sub>t</sub>, A<sub>t</sub>) ← Q(S<sub>t</sub>, A<sub>t</sub>) + α [R + γ max<sub>a</sub> Q(S′, a) − Q(S<sub>t</sub>, A<sub>t</sub>)]',
    note: 'The greedy next action supplies the target while the agent may still explore.',
    code: '<span class="c-comment"># Off-policy Q target</span>\n<span class="c-var">target = reward + gamma * </span>\n<span class="c-var">    np.max(Q[next_state])</span>\n<span class="c-accent">Q[state, action]</span><span class="c-var"> += alpha * (target - Q[state, action])</span>'
  },
  ppo: {
    title: 'Schulman et al. (2017) — PPO Clip Objective', codeTitle: 'Python · Chapter 5',
    formula: 'L<sup>CLIP</sup> = E[min(r<sub>t</sub>(θ)A<sub>t</sub>, clip(r<sub>t</sub>(θ), 1−ε, 1+ε)A<sub>t</sub>)]',
    note: 'Clipping limits destructive policy updates while preserving the useful learning signal.',
    code: '<span class="c-comment"># Clipped policy objective</span>\n<span class="c-var">ratio = torch.exp(new_logp - old_logp)</span>\n<span class="c-accent">clipped = torch.clamp(ratio, 1-eps, 1+eps)</span>\n<span class="c-var">loss = -torch.min(ratio * adv, clipped * adv).mean()</span>'
  }
};
document.querySelectorAll('.paper-tab').forEach(tab => tab.addEventListener('click', () => {
  const example = paperExamples[tab.dataset.paper];
  document.querySelectorAll('.paper-tab').forEach(item => {
    const active = item === tab;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', active);
  });
  document.getElementById('paper-title').textContent = example.title;
  document.getElementById('paper-code-title').textContent = example.codeTitle;
  document.getElementById('paper-formula').innerHTML = example.formula;
  document.getElementById('paper-note').textContent = example.note;
  document.getElementById('paper-code').innerHTML = example.code;
  trackEvent('paper_example_selected', { example: tab.dataset.paper });
}));

// ── Small Q-Learning playground ─────────────────────────────────────────
const playGrid = document.getElementById('playGrid');
const qTable = Array.from({ length: 36 }, () => [0, 0, 0, 0]);
const actions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const startState = 35;
const goalState = 0;
let playAgent = startState;
let playEpisodes = 0;
let lastReward = null;
let learnedPath = [];
let pathTimer = null;

function nextState(state, action) {
  const row = Math.floor(state / 6), col = state % 6;
  const nextRow = Math.max(0, Math.min(5, row + actions[action][0]));
  const nextCol = Math.max(0, Math.min(5, col + actions[action][1]));
  return nextRow * 6 + nextCol;
}
function bestAction(state) {
  const values = qTable[state];
  const max = Math.max(...values);
  return values.indexOf(max);
}
function chooseAction(state, epsilon) {
  return Math.random() < epsilon ? Math.floor(Math.random() * 4) : bestAction(state);
}
function renderPlayground() {
  playGrid.innerHTML = '';
  qTable.forEach((values, index) => {
    const cell = document.createElement('div');
    const value = Math.max(...values);
    cell.className = 'play-cell' + (index === playAgent ? ' agent' : '') + (index === goalState ? ' goal' : '');
    const intensity = index === goalState ? 1 : Math.min(0.85, 0.08 + Math.max(0, value) * 0.7);
    cell.style.background = index === goalState ? 'var(--primary)' : `rgba(255,183,77,${intensity})`;
    cell.title = index === goalState ? 'Goal' : `State ${index + 1}; highest Q-value ${value.toFixed(2)}`;
    cell.textContent = index === goalState ? '★' : index === playAgent ? '●' : '';
    playGrid.appendChild(cell);
  });
}
function trainPlayground() {
  const alpha = Number(document.getElementById('alphaInput').value);
  const epsilon = Number(document.getElementById('epsilonInput').value);
  let reward = 0;
  for (let episode = 0; episode < 100; episode++) {
    let state = startState;
    for (let stepCount = 0; stepCount < 80 && state !== goalState; stepCount++) {
      const action = chooseAction(state, epsilon);
      const next = nextState(state, action);
      const reachedGoal = next === goalState;
      const moved = next !== state;
      const stepReward = reachedGoal ? 1 : moved ? -0.02 : -0.1;
      const target = stepReward + (reachedGoal ? 0 : 0.95 * Math.max(...qTable[next]));
      qTable[state][action] += alpha * (target - qTable[state][action]);
      state = next;
      reward = stepReward;
    }
  }
  playEpisodes += 100;
  lastReward = reward;
  playAgent = startState;
  learnedPath = getGreedyPath();
  renderPlayground();
  document.getElementById('episodeValue').textContent = playEpisodes;
  document.getElementById('pathValue').textContent = learnedPath.length ? `${learnedPath.length - 1} steps` : 'not found';
  document.getElementById('rewardValue').textContent = lastReward.toFixed(2);
  document.getElementById('labMessage').textContent = 'Training complete. Run the learned path to see the greedy policy.';
  trackEvent('playground_trained', { episodes: playEpisodes, learning_rate: alpha, exploration: epsilon });
}
function getGreedyPath() {
  const path = [startState];
  let state = startState;
  for (let i = 0; i < 40 && state !== goalState; i++) {
    const next = nextState(state, bestAction(state));
    if (path.includes(next)) break;
    path.push(next); state = next;
  }
  return state === goalState ? path : [];
}
function runLearnedPath() {
  if (!learnedPath.length) {
    document.getElementById('labMessage').textContent = 'Train the agent before running its learned path.';
    return;
  }
  clearInterval(pathTimer);
  trackEvent('playground_path_run', { path_length: learnedPath.length - 1 });
  let index = 0;
  pathTimer = setInterval(() => {
    playAgent = learnedPath[index++]; renderPlayground();
    if (index >= learnedPath.length) {
      clearInterval(pathTimer);
      document.getElementById('labMessage').textContent = 'The greedy policy reached the goal.';
    }
  }, 180);
}
function resetPlayground() {
  qTable.forEach(values => values.fill(0)); playAgent = startState; playEpisodes = 0; lastReward = null; learnedPath = [];
  clearInterval(pathTimer); renderPlayground();
  document.getElementById('episodeValue').textContent = '0';
  document.getElementById('pathValue').textContent = '—';
  document.getElementById('rewardValue').textContent = '—';
  document.getElementById('labMessage').textContent = 'Train the agent first, then run the learned path.';
}
document.getElementById('trainButton').addEventListener('click', trainPlayground);
document.getElementById('runButton').addEventListener('click', runLearnedPath);
document.getElementById('resetButton').addEventListener('click', resetPlayground);
['alphaInput', 'epsilonInput'].forEach(id => document.getElementById(id).addEventListener('input', event => {
  document.getElementById(id.replace('Input', 'Value')).textContent = event.target.value;
}));
renderPlayground();

// ── Algorithm Marquee ────────────────────────────────────────────────────
const algos = [
  'Q-Learning','DQN','PPO','SAC','DDPG','GRPO','AlphaGo','AlphaZero',
  'RLHF','AlphaDev','DeepSeek-R1','TD(λ)','MCTS','Actor-Critic',
  'Dexterous Manipulation','Humanoid Locomotion','Bellman Equations',
];
const repeated = [...algos, ...algos];
['marquee1','marquee2'].forEach(id => {
  const el = document.getElementById(id);
  repeated.forEach(a => {
    const span = document.createElement('span');
    span.className = 'marquee-chip';
    span.textContent = a;
    el.appendChild(span);
  });
});

// ── Scroll: frosted header ───────────────────────────────────────────────
const hdr = document.getElementById('site-header');
window.addEventListener('scroll', () => {
  hdr.classList.toggle('scrolled', window.scrollY > 48);
}, { passive: true });

// ── Roadmap part visibility tracking ─────────────────────────────────────
const partObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const label = entry.target.querySelector('.part-label')?.textContent.trim();
    const title = entry.target.querySelector('.part-title')?.textContent.trim();
    trackEvent('roadmap_part_viewed', {
      part_label: label,
      part_title: title,
    });
    partObserver.unobserve(entry.target);
  });
}, { threshold: 0.3 });

document.querySelectorAll('.part-card').forEach(card => partObserver.observe(card));

// ── Roadmap filtering, search, and saved progress ───────────────────────
const roadmapRows = [...document.querySelectorAll('.chapter-row')];
const liveRows = roadmapRows.filter(row => row.dataset.status === 'live');
const progressKey = 'rl-seminal-papers-completed';
let completedChapters = new Set(JSON.parse(localStorage.getItem(progressKey) || '[]'));
function updateProgress() {
  document.getElementById('progressLine').textContent = `${completedChapters.size} of ${liveRows.length} live chapters completed`;
  liveRows.forEach(row => {
    const chapter = row.dataset.chapter;
    const button = row.querySelector('.complete-button');
    const complete = completedChapters.has(chapter);
    row.classList.toggle('is-complete', complete);
    if (button) {
      button.classList.toggle('completed', complete);
      button.textContent = complete ? 'Completed' : 'Done';
      button.setAttribute('aria-pressed', complete);
    }
  });
}
document.querySelectorAll('.complete-button').forEach(button => button.addEventListener('click', () => {
  const chapter = button.closest('.chapter-row').dataset.chapter;
  completedChapters.has(chapter) ? completedChapters.delete(chapter) : completedChapters.add(chapter);
  localStorage.setItem(progressKey, JSON.stringify([...completedChapters]));
  updateProgress();
  trackEvent('chapter_completion_toggled', { chapter: chapter, completed: completedChapters.has(chapter) });
}));
function filterRoadmap() {
  const filter = document.querySelector('.filter-button.active').dataset.filter;
  const query = document.getElementById('chapterSearch').value.toLowerCase().trim();
  roadmapRows.forEach(row => {
    const status = row.dataset.status || 'planned';
    const matchesFilter = filter === 'all' || status === filter;
    const matchesSearch = !query || row.textContent.toLowerCase().includes(query);
    row.classList.toggle('is-hidden', !(matchesFilter && matchesSearch));
  });
}
document.querySelectorAll('.filter-button').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.filter-button').forEach(item => item.classList.remove('active'));
  button.classList.add('active'); filterRoadmap();
  trackEvent('roadmap_filter_selected', { filter: button.dataset.filter });
}));
let searchTimer;
document.getElementById('chapterSearch').addEventListener('input', () => {
  filterRoadmap(); clearTimeout(searchTimer);
  searchTimer = setTimeout(() => trackEvent('roadmap_searched'), 700);
});
updateProgress();

// ── Copyable commands ────────────────────────────────────────────────────
document.querySelectorAll('.copy-button').forEach(button => button.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(button.dataset.copy); } catch (_) {
    const area = document.createElement('textarea'); area.value = button.dataset.copy; document.body.appendChild(area);
    area.select(); document.execCommand('copy'); area.remove();
  }
  const original = button.textContent; button.textContent = 'Copied!';
  trackEvent('command_copied', { command: button.dataset.copy });
  setTimeout(() => { button.textContent = original; }, 1400);
}));

// ── Learning path guidance ──────────────────────────────────────────────
const pathAdvice = {
  beginner: 'Recommended route: Chapter 1 → Chapter 2 → the Q-Learning playground above.',
  deep: 'Recommended route: Chapter 3 DQN → Chapter 4 DDPG → Chapter 5 PPO.',
  research: 'Recommended route: compare the Paper → Code examples, then follow the roadmap as new chapters arrive.'
};
document.querySelectorAll('.path-card').forEach(card => {
  const selectPath = () => {
    document.querySelectorAll('.path-card').forEach(item => item.classList.remove('selected'));
    card.classList.add('selected'); document.getElementById('pathAdvice').textContent = pathAdvice[card.dataset.path];
    trackEvent('learning_path_selected', { path: card.dataset.path });
  };
  card.addEventListener('click', selectPath);
  card.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectPath(); } });
});

// ── Mobile nav ───────────────────────────────────────────────────────────
const hamburger = document.querySelector('.hamburger');
const drawer    = document.getElementById('mobile-drawer');
hamburger.addEventListener('click', () => {
  const open = drawer.classList.toggle('open');
  hamburger.setAttribute('aria-expanded', open);
  if (!open) hamburger.focus();
});
drawer.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => {
    drawer.classList.remove('open');
    hamburger.setAttribute('aria-expanded', 'false');
    hamburger.focus();
  });
});
