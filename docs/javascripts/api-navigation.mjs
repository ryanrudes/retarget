/**
 * Hover recognized API symbols in code for mkdocstrings tooltips; click to jump to API docs.
 */

const scope = typeof __md_scope !== "undefined" ? __md_scope : new URL(".", location);
const INDEX_URL = new URL("javascripts/api-symbols.json", scope).href;
const HIGHLIGHT_NAME = "api-nav-target";

/**
 * api-symbols.json stores root-absolute paths (/api/...). On subpath deploys
 * (e.g. GitHub Pages at /retarget/), prepend __md_scope's directory prefix.
 *
 * @param {string} href
 */
function resolveSiteHref(href) {
  if (!href.startsWith("/")) {
    return href;
  }
  const rootPath = new URL(scope).pathname;
  if (rootPath !== "/" && !href.startsWith(rootPath)) {
    return `${rootPath.replace(/\/$/, "")}${href}`;
  }
  return href;
}

/** @type {{ version?: number, byId: Record<string, string>, byShortName: Record<string, string>, titles?: Record<string, string>, titlesByShortName?: Record<string, string> } | null} */
let index = null;

let hoverRaf = 0;
let hoverGeneration = 0;
/** @type {string} */
let lastHoverKey = "";
/** @type {HTMLElement | null} */
let hoverMark = null;
/** @type {HTMLElement | null} */
let tooltipEl = null;
let tooltipHideTimer = 0;

function invalidateIndex() {
  index = null;
}

async function loadIndex() {
  if (index) {
    return index;
  }
  const response = await fetch(INDEX_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API symbol index: ${response.status}`);
  }
  const data = await response.json();
  if (!data.titles || data.version !== 2) {
    console.warn(
      "api-navigation: api-symbols.json is missing tooltip titles; restart `mkdocs serve` or run `mkdocs build`.",
    );
  }
  index = data;
  return index;
}

function isNavigableCode(target) {
  if (!(target instanceof Element)) {
    return null;
  }
  if (target.closest("a[href]")) {
    return null;
  }
  if (target.closest(".live-code-editor, .cm-editor")) {
    return null;
  }
  return target.closest(".md-typeset pre code, .md-typeset code");
}

function caretRangeFromPoint(x, y) {
  let range = document.caretRangeFromPoint?.(x, y) ?? null;
  if (!range && document.caretPositionFromPoint) {
    const pos = document.caretPositionFromPoint(x, y);
    if (pos) {
      range = document.createRange();
      range.setStart(pos.offsetNode, pos.offset);
      range.collapse(true);
    }
  }
  return range;
}

function caretOffsetInElement(element, range) {
  const probe = range.cloneRange();
  probe.selectNodeContents(element);
  probe.setEnd(range.startContainer, range.startOffset);
  return probe.toString().length;
}

/**
 * @param {string} text
 * @param {number} offset
 * @returns {{ symbol: string, start: number, end: number } | null}
 */
function identifierSpanAtOffset(text, offset) {
  const pattern = /[A-Za-z_][\w.]*/g;
  let match = pattern.exec(text);
  while (match) {
    const start = match.index;
    const end = start + match[0].length;
    // Half-open [start, end): caret snapped past a token (e.g. EOL padding) must not match.
    if (offset >= start && offset < end) {
      const symbol = match[0].replace(/\.+$/, "");
      return { symbol, start, end: start + symbol.length };
    }
    match = pattern.exec(text);
  }
  return null;
}

/**
 * @param {Range} symbolRange
 * @param {number} x
 * @param {number} y
 */
function pointerOverSymbolRange(symbolRange, x, y) {
  const pad = 1;
  for (const rect of symbolRange.getClientRects()) {
    if (
      x >= rect.left - pad &&
      x <= rect.right + pad &&
      y >= rect.top - pad &&
      y <= rect.bottom + pad
    ) {
      return true;
    }
  }
  return false;
}

/**
 * @param {HTMLElement} codeEl
 * @param {number} startOffset
 * @param {number} endOffset
 * @returns {Range | null}
 */
function rangeForOffsets(codeEl, startOffset, endOffset) {
  const range = document.createRange();
  const walker = document.createTreeWalker(codeEl, NodeFilter.SHOW_TEXT);
  let charIndex = 0;
  let startNode = null;
  let startOff = 0;
  let endNode = null;
  let endOff = 0;

  while (walker.nextNode()) {
    const node = walker.currentNode;
    const length = node.textContent?.length ?? 0;

    if (!startNode && charIndex + length > startOffset) {
      startNode = node;
      startOff = startOffset - charIndex;
    }
    if (!endNode && charIndex + length >= endOffset) {
      endNode = node;
      endOff = endOffset - charIndex;
      break;
    }
    charIndex += length;
  }

  if (!startNode || !endNode) {
    return null;
  }

  range.setStart(startNode, startOff);
  range.setEnd(endNode, endOff);
  return range;
}

/**
 * @param {string} symbol
 * @param {Awaited<ReturnType<typeof loadIndex>>} data
 */
function hrefForSegment(symbol, data) {
  return data.byId[symbol] ?? data.byShortName[symbol] ?? null;
}

/**
 * Resolve API links for dotted tokens (e.g. fmt.joint_names → joint_names).
 *
 * @param {string} symbol
 * @param {number} start
 * @param {number} end
 * @param {number} offset
 * @param {Awaited<ReturnType<typeof loadIndex>>} data
 * @returns {{ symbol: string, start: number, end: number, href: string } | null}
 */
function resolveSymbolMatch(symbol, start, end, offset, data) {
  const direct = hrefForSegment(symbol, data);
  if (direct) {
    return { symbol, start, end, href: direct };
  }

  if (!symbol.includes(".")) {
    return null;
  }

  const parts = symbol.split(".");
  /** @type {{ part: string, start: number, end: number }[]} */
  const segments = [];
  let pos = start;
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    segments.push({ part, start: pos, end: pos + part.length });
    pos += part.length + (i < parts.length - 1 ? 1 : 0);
  }

  const underCursor = segments.find(
    (seg) => offset >= seg.start && offset < seg.end,
  );
  if (underCursor) {
    const href = hrefForSegment(underCursor.part, data);
    if (href) {
      return {
        symbol: underCursor.part,
        start: underCursor.start,
        end: underCursor.end,
        href,
      };
    }
  }

  // Attribute / registry access: fmt.joint_names, motion_formats.get, result.save_npz
  for (let i = segments.length - 1; i >= 0; i--) {
    const href = hrefForSegment(segments[i].part, data);
    if (href) {
      return {
        symbol: segments[i].part,
        start: segments[i].start,
        end: segments[i].end,
        href,
      };
    }
  }

  return null;
}

function anchorIdFromHref(href) {
  const url = new URL(resolveSiteHref(href), scope);
  const hash = url.hash.slice(1);
  return hash ? decodeURIComponent(hash) : null;
}

/**
 * @param {string | null} anchorId
 * @param {string} symbol
 */
function inferSymbolKind(anchorId, symbol) {
  const name = symbol || anchorId?.split(".").pop() || "";
  if (!name) {
    return "class";
  }
  if (/^[A-Z]/.test(name) && !name.includes("_")) {
    return "class";
  }
  if (anchorId) {
    const parts = anchorId.split(".");
    const parent = parts.length > 1 ? parts[parts.length - 2] : "";
    if (/^[A-Z]/.test(parent) && /^[a-z_]/.test(name)) {
      return "method";
    }
  }
  if (name.startsWith("__") && name.endsWith("__")) {
    return "attribute";
  }
  return "function";
}

/**
 * Same HTML mkdocstrings puts on inline autoref `title` attributes.
 *
 * @param {string | null} anchorId
 * @param {string} symbol
 */
function synthesizeTitleHtml(anchorId, symbol) {
  const name = symbol || anchorId?.split(".").pop() || "";
  if (!name) {
    return null;
  }
  const kind = inferSymbolKind(anchorId, symbol);
  return (
    `<code class="doc-symbol doc-symbol-heading doc-symbol-${kind}"></code>` +
    `            <span class="doc doc-object-name doc-${kind}-name">${name}</span>`
  );
}

/**
 * @param {string | null} anchorId
 * @param {string} symbol
 */
function resolveTitleHtml(anchorId, symbol) {
  if (!index) {
    return null;
  }
  if (anchorId && index.titles?.[anchorId]) {
    return index.titles[anchorId];
  }
  if (index.titlesByShortName?.[symbol]) {
    return index.titlesByShortName[symbol];
  }
  return synthesizeTitleHtml(anchorId, symbol);
}

/**
 * Material content.tooltips markup (md-tooltip2), same as inline autorefs.
 *
 * @param {string} titleHtml
 */
function createTooltip2(titleHtml) {
  const el = document.createElement("div");
  el.className = "md-tooltip2 api-nav-tooltip";
  el.setAttribute("role", "tooltip");
  const inner = document.createElement("div");
  inner.className = "md-tooltip2__inner md-typeset";
  // Preserve inter-tag whitespace from mkdocstrings titles (spacer after "class"/"def" badge).
  inner.innerHTML = titleHtml.trim();
  el.appendChild(inner);
  return el;
}

/**
 * Mirror mkdocs-material Vt() / Xe() placement (tooltip on body, document coords).
 *
 * @param {DOMRect} rect
 * @param {HTMLElement} tooltip
 */
function layoutTooltip2(rect, tooltip) {
  const hostX = rect.left + window.scrollX;
  const hostY = rect.top + window.scrollY;

  tooltip.classList.remove("md-tooltip2--top");
  if (!tooltip.isConnected) {
    document.body.appendChild(tooltip);
  }

  tooltip.style.setProperty("--md-tooltip-host-x", `${hostX}px`);
  tooltip.style.setProperty("--md-tooltip-host-y", `${hostY}px`);

  const inner = tooltip.querySelector(".md-tooltip2__inner");
  const tipWidth = inner instanceof HTMLElement ? inner.offsetWidth : tooltip.offsetWidth;

  // mkdocs-material Xe() / role="tooltip": always below the host, never viewport-flipped.
  const tipX = rect.width / 2;
  const tipY = 8 + rect.height;

  tooltip.classList.remove("md-tooltip2--top");
  tooltip.classList.add("md-tooltip2--bottom");

  tooltip.style.setProperty("--md-tooltip-x", `${tipX}px`);
  tooltip.style.setProperty("--md-tooltip-y", `${tipY}px`);
  tooltip.style.setProperty("--md-tooltip-width", `${tipWidth}px`);
  tooltip.style.setProperty("--md-tooltip-tail", "0px");
}

function clearCodeTooltip() {
  if (!tooltipEl) {
    return;
  }
  tooltipEl.classList.remove("md-tooltip2--active");
  window.clearTimeout(tooltipHideTimer);
  tooltipHideTimer = window.setTimeout(() => {
    tooltipEl?.remove();
    tooltipEl = null;
  }, 250);
}

/**
 * @param {string} titleHtml
 * @param {Range} symbolRange
 */
function showCodeTooltip(titleHtml, symbolRange) {
  if (!titleHtml) {
    clearCodeTooltip();
    return;
  }

  window.clearTimeout(tooltipHideTimer);
  const rect = symbolRange.getBoundingClientRect();
  const innerHtml = titleHtml.trim();
  const needsNew = !tooltipEl || tooltipEl.dataset.apiNavTitle !== innerHtml;

  if (needsNew) {
    tooltipEl?.remove();
    tooltipEl = createTooltip2(innerHtml);
    tooltipEl.dataset.apiNavTitle = innerHtml;
  }

  layoutTooltip2(rect, tooltipEl);
  tooltipEl.classList.add("md-tooltip2--active");

  const inner = tooltipEl.querySelector(".md-tooltip2__inner");
  if (inner instanceof HTMLElement) {
    tooltipEl.style.setProperty("--md-tooltip-width", `${inner.offsetWidth}px`);
  }
}

function clearHoverHighlight() {
  if (typeof CSS !== "undefined" && CSS.highlights) {
    CSS.highlights.delete(HIGHLIGHT_NAME);
  }
  if (hoverMark) {
    const parent = hoverMark.parentNode;
    if (parent) {
      while (hoverMark.firstChild) {
        parent.insertBefore(hoverMark.firstChild, hoverMark);
      }
      hoverMark.remove();
    }
    hoverMark = null;
  }
  lastHoverKey = "";
  document.body.classList.remove("api-nav-hovering");
  clearCodeTooltip();
}

/**
 * @param {Range} range
 */
function applyHoverHighlight(range) {
  if (typeof CSS !== "undefined" && CSS.highlights) {
    CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(range));
    return;
  }

  const mark = document.createElement("mark");
  mark.className = "api-nav-hover-mark";
  try {
    range.surroundContents(mark);
    hoverMark = mark;
  } catch {
    // Shiki layout edge case — hover styling skipped without Highlight API.
  }
}

function navigateTo(href) {
  const url = new URL(resolveSiteHref(href), scope);
  if (url.origin === location.origin && url.pathname === location.pathname && url.hash) {
    const target = document.getElementById(url.hash.slice(1));
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.pushState(null, "", url.hash);
      return;
    }
  }
  location.assign(url.pathname + url.hash + url.search);
}

/**
 * @param {number} x
 * @param {number} y
 */
async function resolveTargetAtPoint(x, y) {
  let code = isNavigableCode(document.elementFromPoint(x, y));
  if (!code) {
    for (const candidate of document.querySelectorAll(".md-typeset pre code")) {
      const box = candidate.getBoundingClientRect();
      if (x >= box.left && x <= box.right && y >= box.top && y <= box.bottom) {
        code = candidate;
        break;
      }
    }
  }
  if (!code) {
    return null;
  }

  const caretRange = caretRangeFromPoint(x, y);
  if (!caretRange || !code.contains(caretRange.startContainer)) {
    return null;
  }

  let data;
  try {
    data = await loadIndex();
  } catch (error) {
    console.warn("api-navigation: failed to load symbol index", error);
    return null;
  }
  const offset = caretOffsetInElement(code, caretRange);
  const span = identifierSpanAtOffset(code.textContent ?? "", offset);
  if (!span) {
    return null;
  }

  const resolved = resolveSymbolMatch(
    span.symbol,
    span.start,
    span.end,
    offset,
    data,
  );
  if (!resolved) {
    return null;
  }

  const symbolRange = rangeForOffsets(code, resolved.start, resolved.end);
  if (!symbolRange || !pointerOverSymbolRange(symbolRange, x, y)) {
    return null;
  }

  const anchorId = anchorIdFromHref(resolved.href);
  const titleHtml = resolveTitleHtml(anchorId, resolved.symbol);

  return {
    code,
    span: {
      symbol: resolved.symbol,
      start: resolved.start,
      end: resolved.end,
    },
    href: resolved.href,
    symbolRange,
    anchorId,
    titleHtml,
  };
}

/**
 * @param {MouseEvent} event
 */
function scheduleHoverUpdate(event) {
  const x = event.clientX;
  const y = event.clientY;

  if (hoverRaf) {
    cancelAnimationFrame(hoverRaf);
  }

  hoverRaf = requestAnimationFrame(async () => {
    hoverRaf = 0;
    const generation = ++hoverGeneration;

    const target = await resolveTargetAtPoint(x, y);
    if (generation !== hoverGeneration) {
      return;
    }
    if (!target) {
      clearHoverHighlight();
      return;
    }

    const hoverKey = `${target.href}@${target.span.start}:${target.span.end}`;
    const inPre = Boolean(target.code.closest("pre"));

    if (hoverKey !== lastHoverKey) {
      if (lastHoverKey) {
        clearHoverHighlight();
      }
      lastHoverKey = hoverKey;
      applyHoverHighlight(target.symbolRange);
    }

    document.body.classList.add("api-nav-hovering");
    if (inPre && target.titleHtml) {
      showCodeTooltip(target.titleHtml, target.symbolRange);
    }
  });
}

function bindHoverHandlers() {
  document.addEventListener("mousemove", scheduleHoverUpdate, true);
  document.addEventListener("mouseleave", clearHoverHighlight, true);
  window.addEventListener("blur", clearHoverHighlight);
}

async function onCodeClick(event) {
  const code = isNavigableCode(event.target);
  if (!code) {
    return;
  }

  const target = await resolveTargetAtPoint(event.clientX, event.clientY);
  if (!target) {
    return;
  }

  event.preventDefault();
  event.stopPropagation();
  navigateTo(target.href);
}

let initialized = false;

function init() {
  if (initialized) {
    invalidateIndex();
    void loadIndex();
    return;
  }
  initialized = true;
  bindHoverHandlers();
  document.addEventListener("click", onCodeClick, true);
  void loadIndex();
}

document$.subscribe(init);
