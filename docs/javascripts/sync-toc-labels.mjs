/**
 * After instant navigation, align sidebar TOC badges with heading decorator labels.
 * Build-time hook patches static HTML; this covers any unpatched or dynamic edge cases.
 */

function syncTocLabelsFromHeadings(root = document) {
  root.querySelectorAll('a.md-nav__link[href^="#"]').forEach((link) => {
    const id = link.getAttribute("href")?.slice(1);
    if (!id) {
      return;
    }
    const heading = document.getElementById(id);
    if (!heading?.classList.contains("doc-heading")) {
      return;
    }
    const label = heading.querySelector(":scope > .doc-labels .doc-label");
    const symbol = link.querySelector("code.doc-symbol-toc");
    if (!label || !symbol) {
      return;
    }
    const clone = label.cloneNode(true);
    clone.classList.add("doc-label-toc");
    symbol.replaceWith(clone);
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    syncTocLabelsFromHeadings();
  });
}
