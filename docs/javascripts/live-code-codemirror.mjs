/**
 * CodeMirror 6 editor for live-code edit mode.
 * Syntax colors come from the same Shiki highlighter as view mode (not Lezer styles).
 */
import { defaultKeymap, indentWithTab } from "https://esm.sh/@codemirror/commands@6.8.1?deps=@codemirror/state@6.5.2,@codemirror/view@6.38.0";
import { python } from "https://esm.sh/@codemirror/lang-python@6.2.0?deps=@codemirror/state@6.5.2,@codemirror/view@6.38.0,@codemirror/language@6.11.0";
import { StreamLanguage } from "https://esm.sh/@codemirror/language@6.11.0?deps=@codemirror/state@6.5.2,@codemirror/view@6.38.0";
import { shell } from "https://esm.sh/@codemirror/legacy-modes@6.5.0/mode/shell?deps=@codemirror/state@6.5.2,@codemirror/language@6.11.0";
import { EditorState } from "https://esm.sh/@codemirror/state@6.5.2";
import { Decoration, EditorView, ViewPlugin, keymap } from "https://esm.sh/@codemirror/view@6.38.0?deps=@codemirror/state@6.5.2";
import {
  THEME_NAME,
  ensureLanguage,
  getHighlighter,
  resolveLanguageId,
} from "./shiki-highlight.mjs";

const sopEditorTheme = EditorView.theme(
  {
    "&": {
      backgroundColor: "transparent",
      color: "#ffffff",
    },
    ".cm-scroller": {
      fontFamily: "inherit",
      fontSize: "inherit",
      lineHeight: "inherit",
      overflow: "auto",
    },
    ".cm-line": {
      fontSize: "inherit",
      lineHeight: "inherit",
      padding: "0",
    },
    ".cm-activeLine": {
      padding: "0",
    },
    ".cm-content": {
      padding: "0",
      caretColor: "#fad000",
    },
    ".cm-cursor": {
      borderLeftColor: "#fad000",
    },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
      backgroundColor: "#b362ff55",
    },
  },
  { dark: true },
);

/** @param {import('shiki').ThemedToken} token */
function tokenAttributes(token) {
  const style = [];
  if (token.color) {
    style.push(`color: ${token.color}`);
  }
  if (token.bgColor) {
    style.push(`background-color: ${token.bgColor}`);
  }
  const fontStyle = token.fontStyle ?? 0;
  if (fontStyle & 1) {
    style.push("font-style: italic");
  }
  if (fontStyle & 2) {
    style.push("font-weight: bold");
  }
  if (fontStyle & 4) {
    style.push("text-decoration: underline");
  }
  if (!style.length) {
    return null;
  }
  return { style: style.join("; ") };
}

/**
 * Paint Shiki token colors as CodeMirror decorations (same engine as view mode).
 * @param {string} lang
 */
function shikiSyntaxHighlight(lang) {
  const languageId = resolveLanguageId(lang);

  return ViewPlugin.fromClass(
    class {
      /** @type {import('@codemirror/view').DecorationSet} */
      decorations = Decoration.none;

      /** @type {import('@codemirror/view').EditorView | null} */
      view = null;

      /** @type {number} */
      requestId = 0;

      /** @type {ReturnType<typeof setTimeout> | null} */
      debounceTimer = null;

      /** @param {import('@codemirror/view').EditorView} view */
      constructor(view) {
        this.view = view;
        this.scheduleHighlight(0);
      }

      /** @param {import('@codemirror/view').ViewUpdate} update */
      update(update) {
        if (update.docChanged) {
          this.scheduleHighlight(60);
        }
      }

      destroy() {
        if (this.debounceTimer !== null) {
          clearTimeout(this.debounceTimer);
        }
        this.requestId += 1;
      }

      /** @param {number} delayMs */
      scheduleHighlight(delayMs) {
        if (this.debounceTimer !== null) {
          clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => {
          this.debounceTimer = null;
          void this.runHighlight();
        }, delayMs);
      }

      async runHighlight() {
        const view = this.view;
        if (!view) {
          return;
        }

        const requestId = ++this.requestId;
        const doc = view.state.doc;
        const code = doc.toString();

        if (!code) {
          this.decorations = Decoration.none;
          view.requestMeasure();
          return;
        }

        try {
          const highlighter = await getHighlighter();
          const langId = await ensureLanguage(highlighter, languageId);
          if (requestId !== this.requestId) {
            return;
          }

          const { tokens } = highlighter.codeToTokens(code, {
            lang: langId,
            theme: THEME_NAME,
          });
          if (requestId !== this.requestId) {
            return;
          }

          const marks = [];
          const lineCount = Math.min(doc.lines, tokens.length);
          for (let lineNo = 1; lineNo <= lineCount; lineNo += 1) {
            const line = doc.line(lineNo);
            const lineTokens = tokens[lineNo - 1];
            let col = line.from;
            for (const token of lineTokens) {
              const len = token.content.length;
              const end = col + len;
              const attrs = tokenAttributes(token);
              if (attrs) {
                marks.push(Decoration.mark({ attributes: attrs }).range(col, end));
              }
              col = end;
            }
          }

          if (requestId !== this.requestId) {
            return;
          }
          this.decorations = Decoration.set(marks, true);
          view.requestMeasure();
        } catch (error) {
          console.error("Shiki edit-mode highlight failed:", error);
        }
      }
    },
    { decorations: (plugin) => plugin.decorations },
  );
}

/** @param {string} lang */
function languageExtension(lang) {
  if (lang === "python" || lang === "py") {
    return python();
  }
  if (lang === "bash" || lang === "sh" || lang === "shell" || lang === "zsh" || lang === "console") {
    return StreamLanguage.define(shell);
  }
  return null;
}

/**
 * @param {HTMLElement} parent
 * @param {string} source
 * @param {string} lang
 * @returns {import('@codemirror/view').EditorView}
 */
export function createLiveEditor(parent, source, lang = "python") {
  parent.innerHTML = "";
  const langExt = languageExtension(lang);
  const extensions = [
    sopEditorTheme,
    shikiSyntaxHighlight(lang),
    EditorView.lineWrapping,
    EditorView.contentAttributes.of({ spellcheck: "false" }),
    keymap.of([...defaultKeymap, indentWithTab]),
  ];
  if (langExt) {
    extensions.unshift(langExt);
  }
  const state = EditorState.create({ doc: source, extensions });
  return new EditorView({ state, parent });
}
