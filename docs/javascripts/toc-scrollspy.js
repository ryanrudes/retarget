/**
 * Fix TOC active indicator for short bottom sections Material never activates
 * (heading cannot scroll up to the sticky offset line).
 */
(function () {
  const BOTTOM_EPS = 8;
  const VIEWPORT_LINE_RATIO = 0.33;

  /** @type {number | null} */
  let pinnedIndex = null;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let pinTimer;

  function activationOffset() {
    const header = document.querySelector(".md-header");
    const main = document.querySelector("main.md-main, .md-main");
    const first = main?.querySelector(":scope > :first-child");
    const headerH = header?.getBoundingClientRect().height ?? 0;
    if (!main || !first) return headerH + 16;
    return headerH + 0.8 * (first.offsetTop - main.offsetTop);
  }

  function isAtPageBottom() {
    return (
      window.scrollY + window.innerHeight >=
      document.documentElement.scrollHeight - BOTTOM_EPS
    );
  }

  /**
   * @param {{ link: HTMLAnchorElement, el: HTMLElement }[]} entries
   */
  function activeIndex(entries) {
    if (!entries.length) return -1;
    if (pinnedIndex !== null && pinnedIndex < entries.length) {
      return pinnedIndex;
    }

    const offset = activationOffset();
    const line = window.scrollY + offset;
    const viewportLine = window.scrollY + window.innerHeight * VIEWPORT_LINE_RATIO;

    if (isAtPageBottom()) {
      return entries.length - 1;
    }

    let passed = 0;
    for (let i = 0; i < entries.length; i++) {
      const top = entries[i].el.offsetTop;
      if (top <= line || top <= viewportLine) {
        passed = i;
      }
    }

    // Short last section: visible at the bottom but its heading never reaches the top line.
    const last = entries.length - 1;
    if (last > passed) {
      const lastRect = entries[last].el.getBoundingClientRect();
      const lastVisible =
        lastRect.top < window.innerHeight && lastRect.bottom > offset;
      if (lastVisible && (isAtPageBottom() || lastRect.top <= offset)) {
        return last;
      }
    }

    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (hash) {
      const idx = entries.findIndex((e) => e.el.id === hash);
      if (idx >= 0) {
        const rect = entries[idx].el.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
          return idx;
        }
      }
    }

    return passed;
  }

  /**
   * @param {HTMLElement} nav
   */
  function collectEntries(nav) {
    /** @type {{ link: HTMLAnchorElement, el: HTMLElement }[]} */
    const entries = [];
    for (const link of nav.querySelectorAll(
      ":scope > .md-nav__list > .md-nav__item > .md-nav__link[href^='#']"
    )) {
      const id = decodeURIComponent(link.hash.slice(1));
      if (!id) continue;
      const el = document.getElementById(id);
      if (el) entries.push({ link, el });
    }
    return entries;
  }

  /**
   * @param {HTMLElement} nav
   * @param {{ link: HTMLAnchorElement, el: HTMLElement }[]} entries
   */
  function applyActive(nav, entries) {
    const idx = activeIndex(entries);
    const links = nav.querySelectorAll(
      ":scope > .md-nav__list > .md-nav__item > .md-nav__link"
    );
    for (const link of links) {
      link.classList.remove("md-nav__link--passed", "md-nav__link--active");
    }
    if (idx < 0) return;
    for (let i = 0; i <= idx && i < entries.length; i++) {
      entries[i].link.classList.add("md-nav__link--passed");
      if (i === idx) {
        entries[i].link.classList.add("md-nav__link--active");
      }
    }
  }

  function pinActive(index) {
    pinnedIndex = index;
    clearTimeout(pinTimer);
    pinTimer = setTimeout(() => {
      pinnedIndex = null;
      updateAll();
    }, 1200);
  }

  function updateAll() {
    for (const nav of document.querySelectorAll(
      ".md-sidebar--secondary .md-nav--secondary, .md-nav--primary .md-nav__item--active > .md-nav--secondary"
    )) {
      if (nav.hidden) continue;
      const entries = collectEntries(nav);
      applyActive(nav, entries);
    }
  }

  function scheduleUpdate() {
    requestAnimationFrame(() => requestAnimationFrame(updateAll));
  }

  function onTocClick(event) {
    const link = event.target.closest(
      ".md-nav--secondary > .md-nav__list > .md-nav__item > .md-nav__link[href^='#']"
    );
    if (!link) return;
    const nav = link.closest(".md-nav--secondary");
    if (!nav || nav.hidden) return;
    const entries = collectEntries(nav);
    const index = entries.findIndex((e) => e.link === link);
    if (index < 0) return;
    pinActive(index);
    scheduleUpdate();
  }

  function setup() {
    updateAll();
  }

  document.addEventListener("click", onTocClick);
  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate, { passive: true });
  window.addEventListener("hashchange", scheduleUpdate);

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => {
      pinnedIndex = null;
      clearTimeout(pinTimer);
      setup();
    });
  } else {
    document.addEventListener("DOMContentLoaded", setup);
  }
})();
