/* Collapsible in-page TOC sections for API and long docs pages. */
(function () {
  const scope = typeof __md_scope !== "undefined" ? __md_scope : new URL(".", location);
  const script = document.createElement("script");
  script.type = "module";
  script.src = new URL("javascripts/toc-collapsible.mjs", scope).href;
  document.head.appendChild(script);
})();
