/**
 * Hide TOC chrome when a page has no in-page headings (empty .md-nav__list).
 * Uses the native `hidden` attribute so Material's layout rules apply.
 */
(function () {
  function tocHasEntries(nav) {
    return Boolean(nav?.querySelector(".md-nav__list > .md-nav__item"));
  }

  function updateEmptyToc() {
    for (const sidebar of document.querySelectorAll(
      '.md-sidebar--secondary[data-md-type="toc"]'
    )) {
      const nav = sidebar.querySelector(".md-nav--secondary");
      sidebar.toggleAttribute("hidden", !tocHasEntries(nav));
    }

    for (const item of document.querySelectorAll(
      ".md-nav--primary > .md-nav__list > .md-nav__item--active"
    )) {
      const nestedNav = item.querySelector(":scope > .md-nav--secondary");
      const hasEntries = tocHasEntries(nestedNav);

      nestedNav?.toggleAttribute("hidden", !hasEntries);

      const tocToggle = item.querySelector(":scope > #__toc");
      const tocLabel = item.querySelector(':scope > label[for="__toc"]');
      tocToggle?.toggleAttribute("hidden", !hasEntries);
      tocLabel?.toggleAttribute("hidden", !hasEntries);
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(updateEmptyToc);
  }
  document.addEventListener("DOMContentLoaded", updateEmptyToc);
})();
