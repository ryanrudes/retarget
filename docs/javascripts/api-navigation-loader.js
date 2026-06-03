/* Loads API symbol hover tooltips and click-to-definition in code blocks. */
(function () {
  const scope = typeof __md_scope !== "undefined" ? __md_scope : new URL(".", location);
  const script = document.createElement("script");
  script.type = "module";
  script.src = new URL("javascripts/api-navigation.mjs", scope).href;
  document.head.appendChild(script);
})();
