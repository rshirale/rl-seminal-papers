// Drives the companion site in headless Chrome and checks that it works.
//
// Run via `make site-check`, which serves docs/ locally first, or point it at
// any deployment:
//
//   SITE_URL=https://rshirale.github.io/rl-seminal-papers/ node tools/site-check/check.mjs
//
// Chrome is found at the usual install path for the platform; set CHROME to
// override it. Set SHOTS=dir to save full-page screenshots there.
//
// The checks are about consistency rather than fixed values: the number of
// live rows must agree with the progress text, the three <head> descriptions,
// the CTA line and llms.txt, so landing a chapter does not mean editing this
// file. The one run most worth keeping is the last page section, which loads
// the site with localStorage throwing -- a private window, or blocked site
// data -- because a single unguarded access there once took half the page
// down without a visible error.

import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const BASE = (process.env.SITE_URL || 'http://127.0.0.1:4173/').replace(/\/?$/, '/');
const SHOTS = process.env.SHOTS;

const CHROME_PATHS = {
  darwin: ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
           '/Applications/Chromium.app/Contents/MacOS/Chromium'],
  linux: ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'],
  win32: ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
          'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'],
};
const chrome = process.env.CHROME || (CHROME_PATHS[process.platform] || []).find(p => fs.existsSync(p));
if (!chrome) {
  console.error('No Chrome found. Install Google Chrome, or set CHROME to a Chrome or Chromium binary.');
  process.exit(2);
}

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: Boolean(ok), detail });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const shot = (page, name) => SHOTS && page.screenshot({ path: path.join(SHOTS, name), fullPage: true });
if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });

const browser = await puppeteer.launch({ executablePath: chrome, headless: true });

// Every page records its own errors and failed requests, so a check can say
// which run produced them.
async function openPage({ width = 1280, height = 900, mobile = false, blockStorage = false, context = browser } = {}) {
  const page = await context.newPage();
  await page.setViewport({ width, height, isMobile: mobile, hasTouch: mobile });
  const errors = [], failed = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('requestfailed', r => failed.push(`${r.url()} ${r.failure()?.errorText}`));
  page.on('response', r => { if (r.status() >= 400) failed.push(`${r.url()} ${r.status()}`); });
  if (blockStorage) {
    await page.evaluateOnNewDocument(() => {
      Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('blocked', 'SecurityError'); } });
    });
  }
  return { page, errors, failed };
}

try {
  // ── Desktop ─────────────────────────────────────────────────────────────
  const context = await browser.createBrowserContext();
  await context.overridePermissions(BASE, ['clipboard-read', 'clipboard-write', 'clipboard-sanitized-write']);
  const { page, errors, failed } = await openPage({ context });
  await page.goto(BASE, { waitUntil: 'networkidle0' });

  const head = await page.evaluate(() => {
    const meta = sel => document.querySelector(sel)?.getAttribute('content') || '';
    return {
      title: document.title,
      ogTitle: meta('meta[property="og:title"]'),
      twitterTitle: meta('meta[name="twitter:title"]'),
      descriptions: [meta('meta[name="description"]'), meta('meta[property="og:description"]'), meta('meta[name="twitter:description"]')],
      icon: document.querySelector('link[rel="icon"]')?.getAttribute('href'),
      footer: document.querySelector('.footer-brand .title')?.textContent.trim(),
      cta: document.querySelector('.cta-sub')?.textContent || '',
      heading: document.getElementById('roadmap-heading')?.textContent || '',
    };
  });
  check('title matches og:title, twitter:title and footer',
    head.title.startsWith(head.ogTitle) && head.ogTitle === head.twitterTitle && head.ogTitle === head.footer,
    JSON.stringify([head.title, head.ogTitle, head.twitterTitle, head.footer]));
  check('favicon declared', head.icon);

  check('consent banner shown on first visit', await page.$eval('#consentBanner', e => !e.hidden));
  await page.click('#rejectAnalytics');
  check('consent banner hides on decline', await page.$eval('#consentBanner', e => e.hidden));

  // Roadmap: counts agree everywhere they are written down.
  const rows = await page.$$eval('.chapter-row', rs => rs.map(r => ({
    n: Number(r.dataset.chapter || r.querySelector('.ch-num').textContent.trim()),
    live: r.dataset.status === 'live',
    title: r.querySelector('.ch-title').textContent.trim(),
  })));
  const live = rows.filter(r => r.live).length;
  const range = `1–${live}`;
  check('chapters numbered 1..N in order', rows.every((r, i) => r.n === i + 1), rows.map(r => r.n).join());
  check('live chapters come first', rows.every((r, i) => r.live === (i < live)));
  check(`roadmap heading counts ${rows.length} chapters`, head.heading.includes(`${rows.length} chapters`), head.heading);
  check(`<head> descriptions say Chapters ${range}`, head.descriptions.every(d => d.includes(`Chapters ${range}`)), head.descriptions.join(' | '));
  check(`CTA says Chapters ${range}`, head.cta.includes(`Chapters ${range}`), head.cta);
  const llms = await (await fetch(BASE + 'llms.txt')).text();
  check(`llms.txt says Chapters ${range}`, llms.includes(`Chapters ${range}`));
  check('llms.txt title matches the page', llms.startsWith(`# ${head.ogTitle}\n`), llms.split('\n')[0]);
  check('llms.txt lists every chapter', rows.every(r => llms.includes(`- Chapter ${r.n}:`)));

  const progress = () => page.$eval('#progressLine', e => e.textContent);
  check('progress text on load', (await progress()) === `0 of ${live} live chapters completed`, await progress());
  await page.click(`.chapter-row[data-chapter="${live}"] .complete-button`);
  check('Done button marks a chapter', (await progress()) === `1 of ${live} live chapters completed`
    && (await page.$eval(`.chapter-row[data-chapter="${live}"] .complete-button`, b => b.textContent + b.getAttribute('aria-pressed'))) === 'Completedtrue');

  const visible = () => page.$$eval('.chapter-row', rs => rs.filter(r => !r.classList.contains('is-hidden')).length);
  for (const f of await page.$$eval('.filter-button', bs => bs.map(b => b.dataset.filter))) {
    await page.click(`.filter-button[data-filter="${f}"]`);
    const want = { all: rows.length, live, planned: rows.length - live }[f];
    check(`filter "${f}"`, want === undefined || (await visible()) === want, `${await visible()} visible, want ${want}`);
  }
  await page.click('.filter-button[data-filter="all"]');
  const search = q => page.$eval('#chapterSearch', (e, q) => { e.value = q; e.dispatchEvent(new Event('input')); }, q);
  const unfound = [];
  for (const r of rows) { await search(r.title); if ((await visible()) < 1) unfound.push(r.title); }
  check('search finds every chapter by title', unfound.length === 0, unfound.join(', '));
  await search('zzzz-no-such-chapter');
  check('search with no match hides every row', (await visible()) === 0);
  await search('');
  check('clearing search restores every row', (await visible()) === rows.length);

  // Papers → Code
  for (const t of await page.$$eval('.paper-tab', ts => ts.map(t => t.dataset.paper))) {
    await page.click(`.paper-tab[data-paper="${t}"]`);
    const s = await page.evaluate(() => ({
      title: document.getElementById('paper-title').textContent,
      code: document.getElementById('paper-code').textContent.length,
      active: document.querySelector('.paper-tab.active')?.dataset.paper,
    }));
    check(`Papers → Code tab "${t}"`, s.active === t && s.title && s.code > 20, s.title);
  }

  // Q-learning playground
  await page.click('#resetButton');
  await page.click('#runButton');
  check('playground: run before train warns', (await page.$eval('#labMessage', e => e.textContent)).startsWith('Train the agent before'));
  await page.click('#trainButton');
  await sleep(300);
  const trained = await page.evaluate(() => ({ ep: document.getElementById('episodeValue').textContent, path: document.getElementById('pathValue').textContent }));
  check('playground: trains', Number(trained.ep) > 0 && trained.path.endsWith('steps'), JSON.stringify(trained));
  await page.click('#runButton');
  await page.waitForFunction(() => document.getElementById('labMessage').textContent.includes('reached the goal'), { timeout: 10000 }).catch(() => {});
  check('playground: learned path reaches the goal', (await page.$eval('#labMessage', e => e.textContent)).includes('reached the goal'));
  await page.click('#resetButton');
  check('playground: resets', (await page.$eval('#episodeValue', e => e.textContent)) === '0');
  await page.$eval('#alphaInput', e => { e.value = e.max; e.dispatchEvent(new Event('input')); });
  check('playground: slider updates its label', (await page.$eval('#alphaValue', e => e.textContent)) === (await page.$eval('#alphaInput', e => e.value)));

  const pauseBefore = await page.$eval('#qPauseBtn', e => e.textContent);
  await page.click('#qPauseBtn');
  check('heatmap pause toggles', (await page.$eval('#qPauseBtn', e => e.textContent)) !== pauseBefore);

  for (const p of await page.$$eval('.path-card', cs => cs.map(c => c.dataset.path))) {
    await page.click(`.path-card[data-path="${p}"]`);
    check(`path card "${p}"`, (await page.$eval('#pathAdvice', e => e.textContent)).startsWith('Recommended route'));
  }

  // Copy buttons: each one copies the command printed beside it.
  const mismatched = await page.$$eval('.copy-wrap', ws => ws
    .filter(w => w.querySelector('code')?.textContent.trim() !== w.querySelector('.copy-button')?.dataset.copy)
    .map(w => w.textContent.slice(0, 60)));
  check('every copy button matches its command', mismatched.length === 0, mismatched.join(' ; '));
  const copyButtons = await page.$$('.copy-button');
  const lastCopy = copyButtons[copyButtons.length - 1];
  const want = await lastCopy.evaluate(b => b.dataset.copy);
  await lastCopy.evaluate(b => b.scrollIntoView());
  await lastCopy.click();
  await sleep(100);
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  check('copy writes the clipboard', clip === want && (await lastCopy.evaluate(b => b.textContent)) === 'Copied!', clip);

  const theme = () => page.evaluate(() => document.documentElement.dataset.theme || 'dark');
  const t0 = await theme();
  await page.click('#themeToggle');
  const t1 = await theme();
  check('theme toggles', t0 !== t1, `${t0} → ${t1}`);
  await shot(page, `desktop-${t1}.png`);

  const badAnchors = await page.$$eval('a[href^="#"]', as => as.map(a => a.getAttribute('href')).filter(h => h.length > 1 && !document.querySelector(h)));
  check('in-page anchors resolve', badAnchors.length === 0, badAnchors.join(' '));

  await page.reload({ waitUntil: 'networkidle0' });
  check('theme persists across reload', (await theme()) === t1);
  check('progress persists across reload', (await progress()) === `1 of ${live} live chapters completed`);
  check('consent choice persists across reload', await page.$eval('#consentBanner', e => e.hidden));
  await page.click('#themeToggle');
  await shot(page, `desktop-${t0}.png`);

  check('desktop: no JS errors', errors.length === 0, errors.join(' | '));
  check('desktop: no failed requests', failed.length === 0, failed.join(' | '));

  // ── Phone width ─────────────────────────────────────────────────────────
  {
    const m = await openPage({ width: 375, height: 812, mobile: true });
    await m.page.goto(BASE, { waitUntil: 'networkidle0' });
    const width = await m.page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
    check('mobile: no horizontal scroll', width[0] <= width[1], `scrollWidth ${width[0]} > ${width[1]}`);
    const drawerOpen = () => m.page.$eval('#mobile-drawer', d => d.classList.contains('open'));
    await m.page.click('.hamburger');
    await sleep(400);
    check('mobile: menu opens', await drawerOpen());
    await m.page.click('#mobile-drawer a[href="#roadmap"]');
    await sleep(600);
    check('mobile: menu link closes the menu', !(await drawerOpen()));
    await m.page.click('#rejectAnalytics').catch(() => {});
    await shot(m.page, 'mobile.png');
    check('mobile: no JS errors', m.errors.length === 0, m.errors.join(' | '));
  }

  // ── Storage blocked ─────────────────────────────────────────────────────
  {
    const s = await openPage({ width: 375, height: 812, mobile: true, blockStorage: true });
    const P = s.page;
    await P.goto(BASE, { waitUntil: 'networkidle0' });
    await P.click('#rejectAnalytics');
    check('storage blocked: consent banner dismisses', await P.$eval('#consentBanner', e => e.hidden));
    await P.$eval('.chapter-row[data-status="live"] .complete-button', e => e.click());
    check('storage blocked: Done button counts', (await P.$eval('#progressLine', e => e.textContent)) === `1 of ${live} live chapters completed`);
    await P.$eval('#chapterSearch', (e, q) => { e.value = q; e.dispatchEvent(new Event('input')); }, rows[0].title);
    check('storage blocked: search filters', (await P.$$eval('.chapter-row:not(.is-hidden)', x => x.length)) < rows.length);
    await P.$eval('.path-card', e => e.click());
    check('storage blocked: path card', (await P.$eval('#pathAdvice', e => e.textContent)).startsWith('Recommended route'));
    await P.$eval('.copy-button', e => e.click());
    await sleep(100);
    check('storage blocked: copy button responds', (await P.$eval('.copy-button', e => e.textContent)) === 'Copied!');
    await P.click('.hamburger');
    check('storage blocked: mobile menu opens', await P.$eval('#mobile-drawer', e => e.classList.contains('open')));
    const before = await P.evaluate(() => document.documentElement.dataset.theme || 'dark');
    await P.$eval('#themeToggle', e => e.click());
    check('storage blocked: theme toggles', (await P.evaluate(() => document.documentElement.dataset.theme)) !== before);
    check('storage blocked: no JS errors', s.errors.length === 0, s.errors.join(' | '));

    const q = await openPage({ blockStorage: true });
    await q.page.goto(BASE + 'privacy.html', { waitUntil: 'networkidle0' });
    await Promise.all([
      q.page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 5000 }).catch(() => {}),
      q.page.click('#changeConsent'),
    ]);
    check('storage blocked: privacy "change consent" returns to the site',
      q.page.url() === BASE + 'index.html' && q.errors.length === 0, `${q.page.url()} ${q.errors.join(' | ')}`);
  }

  // ── Other files ─────────────────────────────────────────────────────────
  for (const file of ['privacy.html', 'llms.txt', 'sitemap.xml', 'robots.txt', 'assets/css/site.css', 'assets/js/site.js', head.icon].filter(Boolean)) {
    const r = await fetch(BASE + file);
    check(`GET ${file}`, r.ok, r.status);
  }
  {
    const p = await openPage();
    await p.page.goto(BASE + 'privacy.html', { waitUntil: 'networkidle0' });
    check('privacy: no errors or failed requests', p.errors.length === 0 && p.failed.length === 0, [...p.errors, ...p.failed].join(' | '));
    const local = await p.page.$$eval('a', as => as.map(a => a.getAttribute('href')).filter(h => h && !/^https?:/.test(h)));
    check('privacy: links back to the site', local.includes('index.html'), local.join(' '));
  }

  // ── External links ──────────────────────────────────────────────────────
  const external = [...new Set(await page.$$eval('a[href^="http"]', as => as.map(a => a.href)))];
  const broken = [];
  await Promise.all(external.map(async u => {
    try {
      const r = await fetch(u, { redirect: 'follow', headers: { 'user-agent': 'Mozilla/5.0 site-check' } });
      if (r.status >= 400) broken.push(`${r.status} ${u}`);
    } catch (e) { broken.push(`ERR ${u} ${e.cause?.code || e.message}`); }
  }));
  check(`external links (${external.length}) resolve`, broken.length === 0, '\n      ' + broken.join('\n      '));
} catch (e) {
  check('site-check ran to completion', false, e.stack);
} finally {
  await browser.close();
}

console.log(`Checked ${BASE}\n`);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.ok ? '' : '  → ' + r.detail}`);
const passed = results.filter(r => r.ok).length;
console.log(`\n${passed}/${results.length} passed`);
process.exit(passed === results.length ? 0 : 1);
