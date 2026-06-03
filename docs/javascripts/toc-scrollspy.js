/**
 * TOC scroll-spy for MkDocs Material (replaces navigation.tracking + toc.follow).
 * Uses data-retarget-toc-active (not md-nav__link--active) so Material cannot
 * leave duplicate yellow indicators on TOC links.
 */
(function () {
  const BOTTOM_EPS = 8;
  const PIN_MS = 1400;
  const NEXT_HEADING_HANDOFF_PX = 96;
  const ACTIVE_ATTR = "data-retarget-toc-active";

  const TOC_NAV_SELECTOR =
    ".md-sidebar--secondary .md-nav--secondary, .md-nav--primary .md-nav__item--active > .md-nav--secondary";

  /** @type {number | null} */
  let pinnedIndex = null;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let pinTimer;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let tocScrollTimer;
  /** @type {number} */
  let scrollRaf = 0;
  /** @type {string | null} */
  let activeId = null;
  /** @type {MutationObserver | undefined} */
  let classObserver;
  let reconciling = false;

  function activationOffset() {
    const header = document.querySelector(".md-header");
    const tabs = document.querySelector(".md-tabs");
    const headerH = header?.getBoundingClientRect().height ?? 0;
    const tabsH =
      tabs && getComputedStyle(tabs).display !== "none"
        ? tabs.getBoundingClientRect().height
        : 0;
    return headerH + tabsH + 12;
  }

  function isAtPageBottom() {
    return (
      window.scrollY + window.innerHeight >=
      document.documentElement.scrollHeight - BOTTOM_EPS
    );
  }

  /**
   * @returns {HTMLElement[]}
   */
  function allTocNavs() {
    return [...document.querySelectorAll(TOC_NAV_SELECTOR)];
  }

  /**
   * @returns {HTMLElement[]}
   */
  function visibleTocNavs() {
    return allTocNavs().filter((nav) => !nav.hidden);
  }

  /**
   * @param {HTMLElement[]} navs
   */
  function canonicalNav(navs) {
    return (
      navs.find((nav) => nav.closest(".md-sidebar--secondary")) ?? navs[0]
    );
  }

  /**
   * @param {HTMLElement} nav
   * @returns {{ links: HTMLAnchorElement[], el: HTMLElement }[]}
   */
  function collectEntries(nav) {
    /** @type {{ links: HTMLAnchorElement[], el: HTMLElement }[]} */
    const entries = [];
    /** @type {Map<string, { links: HTMLAnchorElement[], el: HTMLElement }>} */
    const byId = new Map();

    for (const link of nav.querySelectorAll(".md-nav__link")) {
      const href = link.getAttribute("href");
      if (!href || !href.includes("#")) continue;
      const id = decodeURIComponent(link.hash.slice(1));
      if (!id) continue;
      const el = document.getElementById(id);
      if (!el) continue;

      let entry = byId.get(id);
      if (!entry) {
        entry = { links: [], el };
        byId.set(id, entry);
        entries.push(entry);
      }
      entry.links.push(link);
    }

    return entries;
  }

  /**
   * @param {{ links: HTMLAnchorElement[], el: HTMLElement }[]} entries
   */
  function activeIndex(entries) {
    if (!entries.length) return -1;
    if (pinnedIndex !== null && pinnedIndex < entries.length) {
      return pinnedIndex;
    }

    const offset = activationOffset();
    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (hash) {
      let idx = entries.findIndex((e) => e.el.id === hash);
      if (idx < 0) {
        const target = document.getElementById(hash);
        if (target) {
          idx = entries.findIndex((e) => e.el === target);
        }
      }
      if (idx >= 0) {
        const rect = entries[idx].el.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > offset) {
          return idx;
        }
      }
    }

    let active = 0;
    for (let i = 0; i < entries.length; i++) {
      if (entries[i].el.getBoundingClientRect().top <= offset) {
        active = i;
      }
    }

    const next = active + 1;
    if (next < entries.length) {
      const nextTop = entries[next].el.getBoundingClientRect().top;
      if (nextTop > offset && nextTop - offset < NEXT_HEADING_HANDOFF_PX) {
        active = next;
      }
    }

    const last = entries.length - 1;
    if (last > active && isAtPageBottom()) {
      const lastTop = entries[last].el.getBoundingClientRect().top;
      if (lastTop < window.innerHeight) {
        return last;
      }
    }

    return active;
  }

  function stripMaterialTocState(nav) {
    for (const link of nav.querySelectorAll(".md-nav__link")) {
      link.classList.remove("md-nav__link--passed", "md-nav__link--active");
      link.removeAttribute(ACTIVE_ATTR);
    }
  }

  function clearAllTocHighlights() {
    activeId = null;
    for (const nav of allTocNavs()) {
      stripMaterialTocState(nav);
    }
  }

  /**
   * @param {string | null} id
   */
  function applyActiveId(id) {
    clearAllTocHighlights();
    if (!id) return;

    activeId = id;
    for (const nav of allTocNavs()) {
      for (const link of nav.querySelectorAll(".md-nav__link")) {
        const href = link.getAttribute("href");
        if (!href || !href.includes("#")) continue;
        const linkId = decodeURIComponent(link.hash.slice(1));
        if (linkId === id) {
          link.setAttribute(ACTIVE_ATTR, "true");
        }
      }
    }
  }

  function updateAll() {
    const navs = visibleTocNavs();
    if (!navs.length) return;

    const entries = collectEntries(canonicalNav(navs));
    const idx = activeIndex(entries);
    const id = idx >= 0 ? entries[idx]?.el.id : null;
    applyActiveId(id ?? null);
    scheduleTocScrollSync();
  }

  /**
   * @param {{ links: HTMLAnchorElement[], el: HTMLElement }} entry
   */
  function scrollToEntry(entry) {
    const offset = activationOffset();
    const top =
      entry.el.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
  }

  function pinActive(index) {
    pinnedIndex = index;
    clearTimeout(pinTimer);
    pinTimer = setTimeout(() => {
      pinnedIndex = null;
      updateAll();
    }, PIN_MS);
  }

  /**
   * @param {HTMLElement} nav
   */
  function ensureActiveTocVisible(nav) {
    const wrap = nav.closest(".md-sidebar__scrollwrap");
    if (!wrap) {
      return;
    }

    const active = nav.querySelector(`[${ACTIVE_ATTR}="true"]`);
    if (!active) {
      if (window.scrollY < 8) {
        wrap.scrollTop = 0;
      }
      return;
    }

    const title = nav.querySelector(":scope > .md-nav__title");
    const insetTop =
      (title?.getBoundingClientRect().height ?? 0) +
      parseFloat(getComputedStyle(wrap).scrollPaddingTop || "0") +
      6;
    const insetBottom = 8;
    const linkTop = active.offsetTop - wrap.offsetTop;
    const linkBottom = linkTop + active.offsetHeight;
    const viewTop = wrap.scrollTop;
    const viewBottom = viewTop + wrap.clientHeight;

    let target = viewTop;
    if (linkTop < viewTop + insetTop) {
      target = Math.max(0, linkTop - insetTop);
    } else if (linkBottom > viewBottom - insetBottom) {
      target = Math.min(
        wrap.scrollHeight - wrap.clientHeight,
        linkBottom - wrap.clientHeight + insetBottom,
      );
    } else {
      return;
    }

    if (Math.abs(target - wrap.scrollTop) > 1) {
      wrap.scrollTop = target;
    }
  }

  function syncTocScrollPositions() {
    for (const nav of document.querySelectorAll(
      ".md-sidebar--secondary .md-nav--secondary",
    )) {
      if (nav.hidden) continue;
      ensureActiveTocVisible(nav);
    }
  }

  function scheduleTocScrollSync() {
    clearTimeout(tocScrollTimer);
    tocScrollTimer = setTimeout(() => {
      requestAnimationFrame(() => requestAnimationFrame(syncTocScrollPositions));
    }, 32);
  }

  function scheduleUpdate() {
    if (pinnedIndex !== null) return;
    cancelAnimationFrame(scrollRaf);
    scrollRaf = requestAnimationFrame(updateAll);
  }

  function onTocClick(event) {
    const link = event.target.closest(
      ".md-nav--secondary .md-nav__link[href*='#']",
    );
    if (!link) return;

    const nav = link.closest(".md-nav--secondary");
    if (!nav || nav.hidden) return;

    const navs = visibleTocNavs();
    const entries = collectEntries(canonicalNav(navs));
    const id = decodeURIComponent(link.hash.slice(1));
    const index = entries.findIndex((e) => e.el.id === id);
    if (index < 0) return;

    event.preventDefault();
    const entry = entries[index];
    const hash = entry.links[0]?.hash;
    if (hash) {
      history.pushState(null, "", hash);
    }
    scrollToEntry(entry);
    pinActive(index);
    applyActiveId(entry.el.id);
    requestAnimationFrame(() => ensureActiveTocVisible(nav));
  }

  function reconcileMaterialClasses() {
    if (!activeId || reconciling) return;
    reconciling = true;
    try {
      let changed = false;
      for (const nav of allTocNavs()) {
        for (const link of nav.querySelectorAll(".md-nav__link")) {
          if (
            link.classList.contains("md-nav__link--active") ||
            link.classList.contains("md-nav__link--passed")
          ) {
            link.classList.remove(
              "md-nav__link--active",
              "md-nav__link--passed",
            );
            changed = true;
          }
          const href = link.getAttribute("href");
          const linkId =
            href && href.includes("#")
              ? decodeURIComponent(link.hash.slice(1))
              : "";
          const shouldBeActive = linkId === activeId;
          const isActive = link.getAttribute(ACTIVE_ATTR) === "true";
          if (shouldBeActive && !isActive) {
            link.setAttribute(ACTIVE_ATTR, "true");
            changed = true;
          } else if (!shouldBeActive && isActive) {
            link.removeAttribute(ACTIVE_ATTR);
            changed = true;
          }
        }
      }
      if (changed) {
        scheduleTocScrollSync();
      }
    } finally {
      reconciling = false;
    }
  }

  function observeTocClassMutations() {
    classObserver?.disconnect();
    classObserver = new MutationObserver(() => {
      reconcileMaterialClasses();
    });
    for (const nav of allTocNavs()) {
      classObserver.observe(nav, {
        attributes: true,
        attributeFilter: ["class"],
        subtree: true,
      });
    }
  }

  function setup() {
    observeTocClassMutations();
    updateAll();
  }

  document.addEventListener("click", onTocClick);
  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate, { passive: true });
  if ("onscrollend" in window) {
    window.addEventListener("scrollend", updateAll, { passive: true });
  }
  window.addEventListener("hashchange", () => {
    pinnedIndex = null;
    clearTimeout(pinTimer);
    scheduleUpdate();
  });

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
