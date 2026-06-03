/* Loads the live-code module (MkDocs only injects classic scripts). */
(function () {
  const scope = typeof __md_scope !== "undefined" ? __md_scope : new URL(".", location);
  const script = document.createElement("script");
  script.type = "module";
  script.src = new URL("javascripts/live-code.mjs?v=20260530k", scope).href;
  document.head.appendChild(script);
})();
