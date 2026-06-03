/* Sync sidebar TOC badges with API heading decorator labels. */
(function () {
  const scope = typeof __md_scope !== "undefined" ? __md_scope : new URL(".", location);
  const script = document.createElement("script");
  script.type = "module";
  script.src = new URL("javascripts/sync-toc-labels.mjs", scope).href;
  document.head.appendChild(script);
})();
