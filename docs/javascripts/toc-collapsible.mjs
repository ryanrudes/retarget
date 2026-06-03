/**
 * Collapse nested in-page TOC sections (API classes, category groups) by default.
 * Top-level entries stay visible; expand via chevron or when scroll-spy highlights a child.
 */

const CATEGORY_ID =
  /(?:[-_](attributes|functions|methods)|--(parameters|returns|raises))$/i;

const TOC_NAV_SELECTOR =
  ".md-sidebar--secondary .md-nav--secondary, .md-nav--primary .md-nav__item--active > .md-nav--secondary";

/**
 * @param {HTMLElement} item
 * @param {HTMLElement} rootNav
 */
function isTopLevelTocItem(item, rootNav) {
  const rootList = rootNav.querySelector(":scope > .md-nav__list");
  return Boolean(rootList && item.parentElement === rootList);
}

/**
 * @param {string} href
 */
function isCategoryTocLink(href) {
  const hash = href.includes("#") ? href.slice(href.indexOf("#") + 1) : href;
  return CATEGORY_ID.test(decodeURIComponent(hash));
}

/**
 * @param {HTMLElement} item
 */
function setExpanded(item, expanded) {
  item.classList.toggle("retarget-toc-expanded", expanded);
  item.classList.toggle("retarget-toc-collapsed", !expanded);
  const toggle = item.querySelector(":scope > .retarget-toc-toggle");
  toggle?.setAttribute("aria-expanded", expanded ? "true" : "false");
}

/**
 * @param {HTMLElement} nav
 */
export function setupCollapsibleToc(nav) {
  const rootList = nav.querySelector(":scope > .md-nav__list");
  if (!rootList) {
    return;
  }

  for (const item of nav.querySelectorAll(":scope .md-nav__item")) {
    const childNav = item.querySelector(":scope > nav.md-nav");
    if (!childNav || item.dataset.retargetTocCollapsible !== undefined) {
      continue;
    }

    const link = item.querySelector(":scope > a.md-nav__link");
    const href = link?.getAttribute("href") ?? "";
    if (!isTopLevelTocItem(item, nav) && !isCategoryTocLink(href)) {
      continue;
    }

    item.dataset.retargetTocCollapsible = "";
    item.classList.add("retarget-toc-collapsible", "retarget-toc-collapsed");

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "retarget-toc-toggle";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Expand section");
    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      setExpanded(item, !item.classList.contains("retarget-toc-expanded"));
    });

    link?.insertAdjacentElement("beforebegin", toggle);
  }
}

/**
 * @param {HTMLElement} root
 */
export function setupAllCollapsibleTocs(root = document) {
  for (const nav of root.querySelectorAll(TOC_NAV_SELECTOR)) {
    setupCollapsibleToc(nav);
  }
}

/**
 * @param {HTMLElement} nav
 */
export function expandCollapsibleAncestors(nav) {
  const active = nav.querySelector('[data-retarget-toc-active="true"]');
  if (!active) {
    return;
  }

  let item = active.closest(".md-nav__item");
  while (item) {
    if (item.classList.contains("retarget-toc-collapsible")) {
      setExpanded(item, true);
    }
    const parentNav = item.parentElement?.closest("nav.md-nav");
    item = parentNav?.closest(".md-nav__item") ?? null;
  }
}

/**
 * @param {HTMLElement} root
 */
export function expandAllActiveAncestors(root = document) {
  for (const nav of root.querySelectorAll(TOC_NAV_SELECTOR)) {
    expandCollapsibleAncestors(nav);
  }
}

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    setupAllCollapsibleTocs();
    expandAllActiveAncestors();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupAllCollapsibleTocs();
  expandAllActiveAncestors();
});
