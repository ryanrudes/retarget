/**
 * Live code overlays: Shiki view + CodeMirror edit + console output.
 */
import { ansiTextToHtml, hasAnsi } from "./live-code-ansi.mjs";
import { highlightCodeElement } from "./shiki-highlight.mjs";

const PYTHON_LANGS = new Set(["python", "py"]);
const SHELL_LANGS = new Set(["bash", "sh", "shell", "zsh", "console"]);
const RUNNABLE = new Set([...PYTHON_LANGS, ...SHELL_LANGS]);
const STORAGE_KEY = "retarget-live-code";
const KERNEL_ID_KEY = "retarget-live-kernel-id";
const JUPYTER = {
  baseUrl: "http://127.0.0.1:8888/",
  token: "retarget-docs",
  kernelName: "python3",
};

function jupyterUrl(path) {
  const url = new URL(path, JUPYTER.baseUrl);
  if (!url.searchParams.has("token")) {
    url.searchParams.set("token", JUPYTER.token);
  }
  return url.href;
}

const ICONS = {
  edit: "M20.71,7.04C21.1,6.65 21.1,6 20.71,5.63L18.37,3.29C18,2.9 17.35,2.9 16.96,3.29L15.12,5.12L18.87,8.87M3,17.25V21H6.75L17.81,9.93L14.06,6.18L3,17.25Z",
  done: "M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z",
  run: "M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M10,16.5V7.5L16,12L10,16.5Z",
  reset:
    "M12.5,8C9.85,8 7.45,9.43 6.1,11.44L7.9,12.56C8.85,11.11 10.55,10.25 12.5,10.25V5L8,9.25L12.5,13.5V10.25C14.71,10.25 16.5,12.04 16.5,14.25C16.5,16.46 14.71,18.25 12.5,18.25C10.29,18.25 8.5,16.46 8.5,14.25H6.75C6.75,17.43 9.32,20 12.5,20C15.68,20 18.25,17.43 18.25,14.25C18.25,11.07 15.68,8.5 12.5,8.5",
  copy: "M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z",
  check: "M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z",
};

/** @type {import('@jupyterlab/services').Kernel.IKernelConnection | null} */
let activeKernel = null;
/** @type {Promise<import('@jupyterlab/services').Kernel.IKernelConnection> | null} */
let kernelFlight = null;
/** Bumped on reset so abandoned createKernel() runs cannot publish a kernel. */
let kernelConnectGen = 0;

const KERNEL_CONNECT_MS = 25_000;
const EXECUTE_TIMEOUT_MS = 90_000;
/** @type {Promise<typeof import('./live-code-codemirror.mjs')> | null>} */
let editorModulePromise = null;
/** @type {'uninstalled' | 'inactive' | 'live'} */
let connectionState = "inactive";

function isLive() {
  return connectionState === "live";
}

function isShellLang(lang) {
  return SHELL_LANGS.has(lang);
}

function contentRoot() {
  return document.querySelector(".md-content");
}

function isApiBlock(codeEl) {
  return Boolean(codeEl.closest(".doc, .doc-contents, .doc-object"));
}

function blockLanguage(codeEl) {
  const match = /\blanguage-([\w-]+)\b/i.exec(codeEl.className);
  return match?.[1]?.toLowerCase() ?? "";
}

function findRunnableBlocks() {
  const root = contentRoot();
  if (!root) {
    return [];
  }
  return [...root.querySelectorAll("pre > code")].filter((codeEl) => {
    if (isApiBlock(codeEl)) {
      return false;
    }
    if (codeEl.closest(".live-code, [data-live-skip]")) {
      return false;
    }
    return RUNNABLE.has(blockLanguage(codeEl));
  });
}

function getHighlightRoot(codeEl) {
  return codeEl.closest("div.highlight") ?? codeEl.closest("pre");
}

function getViewPre(codeEl) {
  return codeEl.closest("pre");
}

/** @param {HTMLElement} pre */
function getCmMount(pre) {
  let mount = pre.querySelector(":scope > .live-code__cm-mount");
  if (!mount) {
    mount = document.createElement("div");
    mount.className = "live-code__cm-mount";
    pre.appendChild(mount);
  }
  return mount;
}

/** MkDocs Material copy control (not our toolbar copy button). */
function stripMaterialCopyButtons(root) {
  if (!root) {
    return;
  }
  root.querySelectorAll(".md-clipboard, .md-code__button").forEach((el) => {
    if (!el.closest(".live-code__actions")) {
      el.remove();
    }
  });
}

function iconButton(action, label, extraClass = "") {
  const path = ICONS[action];
  return [
    `<button type="button" class="live-code__icon-btn ${extraClass}" data-action="${action}"`,
    ` aria-label="${label}" title="${label}">`,
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true">`,
    `<path fill="currentColor" d="${path}"/></svg></button>`,
  ].join("");
}

async function getEditorModule() {
  if (!editorModulePromise) {
    editorModulePromise = import("./live-code-codemirror.mjs");
  }
  return editorModulePromise;
}

function readSource(wrap, codeEl) {
  if (wrap._cmView) {
    return wrap._cmView.state.doc.toString();
  }
  if (wrap?.dataset.liveOriginal != null) {
    return wrap.dataset.liveOriginal;
  }
  return codeEl.textContent ?? "";
}

function writePlainSource(codeEl, source) {
  const pre = codeEl.closest("pre");
  codeEl.textContent = source;
  codeEl.classList.remove("shiki");
  delete codeEl.dataset.shikiHighlighted;
  pre?.classList.remove("shiki");
}

async function restoreHighlightedSource(codeEl, source) {
  writePlainSource(codeEl, source);
  try {
    await highlightCodeElement(codeEl);
  } catch (error) {
    console.error("Re-highlight failed:", error);
  }
}

async function probeJupyter() {
  try {
    const response = await fetch(jupyterUrl("api"), {
      method: "GET",
      mode: "cors",
    });
    return response.ok;
  } catch {
    return false;
  }
}

function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(message)), ms);
    }),
  ]);
}

function kernelIsUsable(kernel) {
  return (
    kernel.connectionStatus === "connected" &&
    kernel.status !== "dead" &&
    kernel.status !== "terminating"
  );
}

async function waitForKernelConnected(kernel, timeoutMs = KERNEL_CONNECT_MS) {
  if (kernelIsUsable(kernel)) {
    return kernel;
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(
        new Error(
          "Could not connect to the Jupyter kernel. Hard-refresh the docs page, then toggle Live again.",
        ),
      );
    }, timeoutMs);

    const finishIfReady = () => {
      if (kernelIsUsable(kernel)) {
        cleanup();
        resolve(kernel);
      }
    };

    const onStatus = (_, status) => {
      if (status === "connected") {
        finishIfReady();
      } else if (status === "dead") {
        cleanup();
        reject(new Error("Kernel died before connecting."));
      }
    };

    const cleanup = () => {
      clearTimeout(timeout);
      clearInterval(poll);
      kernel.connectionStatusChanged.disconnect(onStatus);
    };

    kernel.connectionStatusChanged.connect(onStatus);
    const poll = setInterval(finishIfReady, 200);
    if (kernel.ready) {
      void kernel.ready.then(finishIfReady).catch(() => undefined);
    }
    finishIfReady();
  });
}

async function disposeKernel(kernel) {
  if (!kernel) {
    return;
  }
  try {
    kernel.dispose();
  } catch {
    /* ignore */
  }
  try {
    await Promise.race([
      kernel.shutdown(),
      new Promise((resolve) => setTimeout(resolve, 2000)),
    ]);
  } catch {
    /* ignore */
  }
}

async function shutdownKernelId(kernelId) {
  if (!kernelId) {
    return;
  }
  const { KernelAPI, ServerConnection } = await import(
    "https://esm.sh/@jupyterlab/services@7.3.6"
  );
  const serverSettings = ServerConnection.makeSettings({
    baseUrl: JUPYTER.baseUrl,
    token: JUPYTER.token,
    appendToken: true,
  });
  try {
    await KernelAPI.shutdownKernel(kernelId, serverSettings);
  } catch {
    /* ignore */
  }
}

async function resetKernel() {
  kernelConnectGen += 1;
  kernelFlight = null;
  const kernel = activeKernel;
  activeKernel = null;
  const storedId = sessionStorage.getItem(KERNEL_ID_KEY);
  sessionStorage.removeItem(KERNEL_ID_KEY);
  await disposeKernel(kernel);
  if (storedId && (!kernel || storedId !== kernel.id)) {
    await shutdownKernelId(storedId);
  }
}

async function createKernel() {
  const gen = kernelConnectGen;
  const { KernelAPI, KernelManager, ServerConnection } = await import(
    "https://esm.sh/@jupyterlab/services@7.3.6"
  );
  const serverSettings = ServerConnection.makeSettings({
    baseUrl: JUPYTER.baseUrl,
    token: JUPYTER.token,
    appendToken: true,
  });
  const response = await ServerConnection.makeRequest(
    `${serverSettings.baseUrl}api`,
    {},
    serverSettings,
  );
  if (response.status !== 200) {
    throw new Error(`Jupyter not reachable (HTTP ${response.status})`);
  }

  const manager = new KernelManager({ serverSettings });
  const storedId = sessionStorage.getItem(KERNEL_ID_KEY);
  let kernel = null;

  if (storedId) {
    const running = await KernelAPI.listRunning(serverSettings);
    const model = running.find((entry) => entry.id === storedId);
    if (model && model.execution_state !== "dead") {
      try {
        kernel = await manager.connectTo({ model });
      } catch {
        await shutdownKernelId(storedId);
        sessionStorage.removeItem(KERNEL_ID_KEY);
      }
    } else {
      sessionStorage.removeItem(KERNEL_ID_KEY);
    }
  }

  if (!kernel) {
    if (storedId) {
      await shutdownKernelId(storedId);
      sessionStorage.removeItem(KERNEL_ID_KEY);
    }
    const model = await KernelAPI.startNew(
      {
        name: JUPYTER.kernelName,
        env: { PYTHONUNBUFFERED: "1" },
      },
      serverSettings,
    );
    kernel = await manager.connectTo({ model });
    sessionStorage.setItem(KERNEL_ID_KEY, kernel.id);
  }

  if (gen !== kernelConnectGen) {
    await disposeKernel(kernel);
    throw new Error("Kernel connect cancelled.");
  }

  await waitForKernelConnected(kernel);

  if (gen !== kernelConnectGen) {
    await disposeKernel(kernel);
    throw new Error("Kernel connect cancelled.");
  }

  activeKernel = kernel;
  return kernel;
}

async function getKernel() {
  if (activeKernel && kernelIsUsable(activeKernel)) {
    return activeKernel;
  }

  if (kernelFlight) {
    return kernelFlight;
  }

  kernelFlight = (async () => {
    await disposeKernel(activeKernel);
    activeKernel = null;
    return createKernel();
  })().finally(() => {
    kernelFlight = null;
  });

  return kernelFlight;
}

function buildShellRunner(code) {
  const payload = JSON.stringify(code);
  return `
import json as _json
import os as _os
import subprocess as _subprocess
import sys as _sys
import time as _time

_CODE = _json.loads(${JSON.stringify(payload)})
_ENV = {
    **_os.environ,
    "TERM": "xterm-256color",
    "COLORTERM": "truecolor",
    "FORCE_COLOR": "1",
    "PYTHONUNBUFFERED": "1",
}


def _run_plain(command):
    result = _subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=_os.getcwd(),
        env=_ENV,
    )
    out = (result.stdout or "") + (result.stderr or "")
    return out, result.returncode


def _emit_bytes(data):
    if not data:
        return
    _sys.stdout.write(data.decode("utf-8", errors="replace"))
    _sys.stdout.flush()


def _drain_master(master):
    import select as _select

    while True:
        ready, _, _ = _select.select([master], [], [], 0)
        if master not in ready:
            break
        data = _os.read(master, 65536)
        if not data:
            break
        _emit_bytes(data)


def _run_pty(command, timeout=120):
    import pty as _pty
    import select as _select

    master, slave = _pty.openpty()
    proc = _subprocess.Popen(
        command,
        shell=True,
        stdout=slave,
        stderr=slave,
        stdin=slave,
        close_fds=True,
        cwd=_os.getcwd(),
        env=_ENV,
    )
    _os.close(slave)
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        ready, _, _ = _select.select([master], [], [], 0.1)
        if master in ready:
            data = _os.read(master, 65536)
            if not data:
                break
            _emit_bytes(data)
        elif proc.poll() is not None:
            _drain_master(master)
            break
    try:
        proc.wait(timeout=5)
    except _subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    _drain_master(master)
    _os.close(master)
    return proc.returncode


try:
    if _sys.platform == "win32":
        _out, _rc = _run_plain(_CODE)
        if _out:
            _sys.stdout.write(_out)
            _sys.stdout.flush()
    else:
        _rc = _run_pty(_CODE)
except Exception as _exc:
    _sys.stderr.write(f"{type(_exc).__name__}: {_exc}\\n")
    _sys.stderr.flush()
    _sys.exit(1)
if _sys.platform == "win32" and _rc != 0:
    _sys.stderr.write(f"\\n(exit code {_rc})\\n")
    _sys.stderr.flush()
elif _sys.platform != "win32" and _rc != 0:
    _sys.stderr.write(f"\\n(exit code {_rc})\\n")
    _sys.stderr.flush()
`;
}

const PYTHON_STREAM_PREAMBLE = `
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
`;

function kernelCode(code, lang) {
  if (isShellLang(lang)) {
    return buildShellRunner(code);
  }
  return `${PYTHON_STREAM_PREAMBLE}${code}`;
}

function appendPlainOutput(buffer, content) {
  const text = content.data?.["text/plain"];
  if (!text) {
    return buffer;
  }
  return buffer + (buffer && !buffer.endsWith("\n") ? "\n" : "") + text;
}

/** @param {Record<string, unknown> | undefined} data */
function mimeToString(data) {
  if (!data) {
    return "";
  }
  const html = data["text/html"];
  if (html) {
    return Array.isArray(html) ? html.join("") : String(html);
  }
  return "";
}

/**
 * @param {HTMLElement} consoleEl
 * @returns {{
 *   appendStdout: (text: string) => void,
 *   appendStderr: (text: string) => void,
 *   appendStdoutHtml: (html: string) => void,
 *   appendStderrHtml: (html: string) => void,
 *   finish: () => { stdout: string, stderr: string, stdoutHtml: string, stderrHtml: string },
 * }}
 */
function createConsoleStream(consoleEl) {
  showConsole(consoleEl);
  consoleEl.innerHTML = '<span class="live-code__muted">Running…</span>';

  const state = {
    stdout: "",
    stderr: "",
    stdoutHtml: "",
    stderrHtml: "",
  };
  let started = false;
  let outPre = null;
  let errPre = null;
  let outRenderScheduled = false;
  let errRenderScheduled = false;

  const outEl = document.createElement("div");
  outEl.className = "live-code__stream-out";
  const errEl = document.createElement("div");
  errEl.className = "live-code__stream-err";

  const scrollToEnd = () => {
    consoleEl.scrollTop = consoleEl.scrollHeight;
  };

  const ensureStarted = () => {
    if (started) {
      return;
    }
    started = true;
    consoleEl.replaceChildren(outEl, errEl);
  };

  const renderOutFormatted = () => {
    if (state.stdoutHtml) {
      outEl.innerHTML = `<div class="live-code__rich live-code__rich--out">${sanitizeRichHtml(state.stdoutHtml)}</div>`;
      outPre = null;
      return;
    }
    if (!state.stdout) {
      outEl.replaceChildren();
      outPre = null;
      return;
    }
    const formatted = stdoutToHtml(state.stdout);
    outEl.innerHTML = formatted.startsWith("<pre")
      ? formatted
      : `<div class="live-code__rich live-code__rich--out">${formatted}</div>`;
    outPre = null;
  };

  const renderErrFormatted = () => {
    if (state.stderrHtml) {
      errEl.innerHTML = `<div class="live-code__rich live-code__rich--err">${sanitizeRichHtml(state.stderrHtml)}</div>`;
      errPre = null;
      return;
    }
    if (!state.stderr) {
      errEl.replaceChildren();
      errPre = null;
      return;
    }
    const formatted = stderrToHtml(state.stderr);
    errEl.innerHTML = formatted.startsWith("<pre")
      ? formatted
      : `<div class="live-code__rich live-code__rich--err">${formatted}</div>`;
    errPre = null;
  };

  const scheduleOutRender = () => {
    if (outRenderScheduled) {
      return;
    }
    outRenderScheduled = true;
    requestAnimationFrame(() => {
      outRenderScheduled = false;
      ensureStarted();
      renderOutFormatted();
      scrollToEnd();
    });
  };

  const scheduleErrRender = () => {
    if (errRenderScheduled) {
      return;
    }
    errRenderScheduled = true;
    requestAnimationFrame(() => {
      errRenderScheduled = false;
      ensureStarted();
      renderErrFormatted();
      scrollToEnd();
    });
  };

  const appendPlainChunk = (el, preRef, text, preClass) => {
    if (!preRef) {
      preRef = document.createElement("pre");
      preRef.className = preClass;
      el.replaceChildren(preRef);
    }
    preRef.append(document.createTextNode(text));
    scrollToEnd();
    return preRef;
  };

  return {
    appendStdout(text) {
      if (!text) {
        return;
      }
      ensureStarted();
      if (state.stdoutHtml) {
        state.stdoutHtml += text;
        scheduleOutRender();
        return;
      }
      state.stdout += text;
      if (hasAnsi(state.stdout)) {
        scheduleOutRender();
        return;
      }
      outPre = appendPlainChunk(outEl, outPre, text, "live-code__out");
    },
    appendStderr(text) {
      if (!text) {
        return;
      }
      ensureStarted();
      if (state.stderrHtml) {
        state.stderrHtml += text;
        scheduleErrRender();
        return;
      }
      state.stderr += text;
      if (hasAnsi(state.stderr)) {
        scheduleErrRender();
        return;
      }
      errPre = appendPlainChunk(errEl, errPre, text, "live-code__err");
    },
    appendStdoutHtml(html) {
      state.stdoutHtml += html;
      scheduleOutRender();
    },
    appendStderrHtml(html) {
      state.stderrHtml += html;
      scheduleErrRender();
    },
    finish() {
      ensureStarted();
      renderOutFormatted();
      renderErrFormatted();
      if (!state.stdout && !state.stderr && !state.stdoutHtml && !state.stderrHtml) {
        consoleEl.innerHTML = '<span class="live-code__muted">(no output)</span>';
      }
      scrollToEnd();
      return { ...state };
    },
  };
}

async function runOnKernel(kernel, code, lang, stream) {
  const future = kernel.requestExecute({ code: kernelCode(code, lang) }, false, {});
  let stdout = "";
  let stderr = "";
  let stdoutHtml = "";
  let stderrHtml = "";

  future.onIOPub = (msg) => {
    const type = msg.header.msg_type;
    const content = msg.content;
    if (type === "stream") {
      if (content.name === "stdout") {
        stdout += content.text;
        stream?.appendStdout(content.text);
      } else if (content.name === "stderr") {
        stderr += content.text;
        stream?.appendStderr(content.text);
      }
    } else if (type === "execute_result" || type === "display_data") {
      const html = mimeToString(content.data);
      if (html) {
        stdoutHtml += html;
        stream?.appendStdoutHtml(html);
      } else {
        stdout = appendPlainOutput(stdout, content);
        stream?.appendStdout(
          content.data?.["text/plain"] ?? "",
        );
      }
    } else if (type === "error") {
      const html = mimeToString(content.data);
      if (html) {
        stderrHtml += html;
        stream?.appendStderrHtml(html);
      } else {
        const text = (content.traceback ?? []).join("\n");
        stderr += text;
        stream?.appendStderr(text);
      }
    }
  };

  const reply = await withTimeout(
    future.done,
    EXECUTE_TIMEOUT_MS,
    "Execution timed out after 90 seconds.",
  );
  if (reply.content.status === "error" && !stderr && !stderrHtml) {
    const html = mimeToString(reply.content.data);
    if (html) {
      stderrHtml += html;
      stream?.appendStderrHtml(html);
    } else {
      const text = (reply.content.traceback ?? []).join("\n");
      stderr += text;
      stream?.appendStderr(text);
    }
  }
  return { stdout, stderr, stdoutHtml, stderrHtml };
}

async function executeCode(code, lang, stream) {
  let lastError;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const kernel = await getKernel();
      return await runOnKernel(kernel, code, lang, stream);
    } catch (error) {
      lastError = error;
      console.warn("Kernel run failed, resetting kernel:", error);
      await resetKernel();
    }
  }
  throw lastError;
}

function showConsole(consoleEl) {
  consoleEl.hidden = false;
  consoleEl.setAttribute("aria-live", "polite");
}

function sanitizeRichHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  template.content
    .querySelectorAll("script, iframe, object, embed, link[rel='stylesheet']")
    .forEach((el) => el.remove());
  return template.innerHTML;
}

/** @param {string} text */
function ansiOutputToHtml(text, preClass) {
  if (!text) {
    return "";
  }
  if (hasAnsi(text)) {
    return ansiTextToHtml(text);
  }
  return `<pre class="${preClass}">${escapeHtml(text)}</pre>`;
}

/** @param {string} stdout */
function stdoutToHtml(stdout) {
  return ansiOutputToHtml(stdout, "live-code__out");
}

/** @param {string} stderr */
function stderrToHtml(stderr) {
  return ansiOutputToHtml(stderr, "live-code__err");
}

async function renderConsole(consoleEl, { stdout, stderr, stdoutHtml, stderrHtml, error }) {
  showConsole(consoleEl);
  const parts = [];
  if (error) {
    parts.push(`<div class="live-code__err">${escapeHtml(error)}</div>`);
  }
  if (stdoutHtml) {
    parts.push(`<div class="live-code__rich live-code__rich--out">${sanitizeRichHtml(stdoutHtml)}</div>`);
  } else if (stdout) {
    const formatted = stdoutToHtml(stdout);
    if (formatted.startsWith("<pre")) {
      parts.push(formatted);
    } else {
      parts.push(`<div class="live-code__rich live-code__rich--out">${formatted}</div>`);
    }
  }
  if (stderrHtml) {
    parts.push(`<div class="live-code__rich live-code__rich--err">${sanitizeRichHtml(stderrHtml)}</div>`);
  } else if (stderr) {
    const formatted = stderrToHtml(stderr);
    if (formatted.startsWith("<pre")) {
      parts.push(formatted);
    } else {
      parts.push(`<div class="live-code__rich">${formatted}</div>`);
    }
  }
  if (!parts.length) {
    parts.push('<span class="live-code__muted">(no output)</span>');
  }
  consoleEl.innerHTML = parts.join("");
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setEditButtonState(editBtn, editing) {
  if (!editBtn) {
    return;
  }
  const label = editing ? "Done editing" : "Edit code";
  editBtn.setAttribute("aria-pressed", editing ? "true" : "false");
  editBtn.title = label;
  editBtn.setAttribute("aria-label", label);
  const path = editBtn.querySelector("path");
  if (path) {
    path.setAttribute("d", editing ? ICONS.done : ICONS.edit);
  }
}

/** @param {{ stdout?: string, stderr?: string, stdoutHtml?: string, stderrHtml?: string, error?: string }} payload */
function runHadFailure(payload) {
  return Boolean(
    payload?.error?.trim() || payload?.stderr?.trim() || payload?.stderrHtml?.trim(),
  );
}

/** @param {HTMLElement} wrap */
function setRunFailed(wrap, failed) {
  wrap.querySelector('[data-action="run"]')?.classList.toggle(
    "live-code__icon-btn--run-failed",
    failed,
  );
}

function setCopyButtonState(copyBtn, copied) {
  if (!copyBtn) {
    return;
  }
  const path = copyBtn.querySelector("path");
  if (!path) {
    return;
  }
  if (copied) {
    copyBtn.classList.add("live-code__icon-btn--copied");
    copyBtn.setAttribute("aria-label", "Copied");
    copyBtn.title = "Copied!";
    path.setAttribute("d", ICONS.check);
    return;
  }
  copyBtn.classList.remove("live-code__icon-btn--copied");
  copyBtn.setAttribute("aria-label", "Copy code");
  copyBtn.title = "Copy code";
  path.setAttribute("d", ICONS.copy);
}

async function destroyEditor(wrap) {
  const codeEl = wrap.querySelector("code");
  if (wrap._cmView) {
    wrap._cmView.destroy();
    wrap._cmView = null;
  }
  wrap.querySelector(".live-code__cm-mount")?.remove();
  if (codeEl) {
    codeEl.classList.remove("live-code__view-sleep");
  }
}

async function setEditing(wrap, editing) {
  const codeEl = wrap.querySelector("code");
  const highlight = getHighlightRoot(codeEl);
  const pre = getViewPre(codeEl);
  const editBtn = wrap.querySelector('[data-action="edit"]');
  const resetBtn = wrap.querySelector('[data-action="reset"]');
  if (!codeEl || !highlight || !pre || !editBtn) {
    return;
  }

  if (editing) {
    wrap.classList.add("live-code--editing");
    setEditButtonState(editBtn, true);
    const source = readSource(wrap, codeEl);
    codeEl.classList.add("live-code__view-sleep");
    const mount = getCmMount(pre);
    try {
      const { createLiveEditor } = await getEditorModule();
      wrap._cmView = createLiveEditor(mount, source, wrap.dataset.liveLang ?? "python");
      wrap._cmView.focus();
    } catch (error) {
      wrap.classList.remove("live-code--editing");
      setEditButtonState(editBtn, false);
      codeEl.classList.remove("live-code__view-sleep");
      mount.remove();
      const consoleEl = wrap.querySelector(".live-code__console");
      const message = error instanceof Error ? error.message : String(error);
      void renderConsole(consoleEl, {
        error: `Editor failed to load: ${message}`,
      });
      return;
    }
    if (resetBtn) {
      resetBtn.disabled = false;
    }
    return;
  }

  wrap.classList.remove("live-code--editing");
  setEditButtonState(editBtn, false);
  const source = readSource(wrap, codeEl);
  await destroyEditor(wrap);
  wrap.dataset.liveOriginal = source;
  await restoreHighlightedSource(codeEl, source);
}

function wrapBlock(codeEl) {
  if (codeEl.closest(".live-code")) {
    return null;
  }

  const highlightRoot = getHighlightRoot(codeEl);
  if (!highlightRoot?.parentNode) {
    return null;
  }

  const lang = blockLanguage(codeEl);
  const source = codeEl.textContent ?? "";
  const wrap = document.createElement("div");
  wrap.className = "live-code";
  wrap.dataset.liveOriginal = source;
  wrap.dataset.liveLang = lang;

  const block = document.createElement("div");
  block.className = "live-code__block";

  const actions = document.createElement("div");
  actions.className = "live-code__actions";
  actions.setAttribute("role", "toolbar");
  actions.setAttribute("aria-label", "Code block actions");
  actions.innerHTML = [
    iconButton("edit", "Edit code"),
    iconButton("run", "Run code", "live-code__icon-btn--run"),
    iconButton("reset", "Reset code"),
    iconButton("copy", "Copy code"),
  ].join("");
  actions.querySelector('[data-action="reset"]').disabled = true;

  const consoleEl = document.createElement("div");
  consoleEl.className = "live-code__console";
  consoleEl.hidden = true;

  const parent = highlightRoot.parentNode;
  if (parent) {
    for (const child of [...parent.children]) {
      if (
        child !== highlightRoot &&
        (child.classList.contains("md-clipboard") || child.classList.contains("md-code__button"))
      ) {
        child.remove();
      }
    }
  }

  parent?.insertBefore(wrap, highlightRoot);
  block.append(actions, highlightRoot);
  wrap.append(block, consoleEl);
  stripMaterialCopyButtons(block);

  actions.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-action]");
    if (!btn || btn.disabled) {
      return;
    }
    void handleAction(wrap, btn.dataset.action);
  });

  return wrap;
}

async function copySource(wrap, codeEl, copyBtn) {
  const text = readSource(wrap, codeEl);
  try {
    await navigator.clipboard.writeText(text);
    setCopyButtonState(copyBtn, true);
    setTimeout(() => setCopyButtonState(copyBtn, false), 1600);
  } catch (error) {
    console.error("Copy failed:", error);
    const consoleEl = wrap.querySelector(".live-code__console");
    void renderConsole(consoleEl, { error: "Copy failed — check browser clipboard permission." });
  }
}

async function handleAction(wrap, action) {
  const codeEl = wrap.querySelector("code");
  const consoleEl = wrap.querySelector(".live-code__console");
  const resetBtn = wrap.querySelector('[data-action="reset"]');
  const lang = wrap.dataset.liveLang ?? blockLanguage(codeEl);
  if (!codeEl || !consoleEl) {
    return;
  }

  if (action === "copy") {
    const copyBtn = wrap.querySelector('[data-action="copy"]');
    await copySource(wrap, codeEl, copyBtn);
    return;
  }

  if (action === "edit") {
    const editing = !wrap.classList.contains("live-code--editing");
    await setEditing(wrap, editing);
    if (!editing) {
      const source = readSource(wrap, codeEl);
      wrap.dataset.liveOriginal = source;
      if (resetBtn) {
        resetBtn.disabled = source === wrap.dataset.liveBaseline;
      }
    }
    return;
  }

  if (action === "reset") {
    if (wrap.classList.contains("live-code--editing")) {
      await setEditing(wrap, false);
    }
    consoleEl.hidden = true;
    consoleEl.innerHTML = "";
    wrap.classList.remove("live-code--ran");
    setRunFailed(wrap, false);
    await restoreHighlightedSource(codeEl, wrap.dataset.liveBaseline ?? "");
    wrap.dataset.liveOriginal = wrap.dataset.liveBaseline ?? "";
    if (resetBtn) {
      resetBtn.disabled = true;
    }
    return;
  }

  if (action === "run") {
    if (!isLive()) {
      void renderConsole(consoleEl, {
        error:
          connectionState === "uninstalled"
            ? "Jupyter is not running — see /interactive-playground/ for setup."
            : "Turn on Live mode using the switch at the bottom-right.",
      });
      setRunFailed(wrap, true);
      return;
    }

    const runBtn = wrap.querySelector('[data-action="run"]');
    runBtn.disabled = true;
    setRunFailed(wrap, false);
    const stream = createConsoleStream(consoleEl);

    let failed = false;
    try {
      await getKernel();
      const code = readSource(wrap, codeEl);
      const result = await executeCode(code, lang, stream);
      stream.finish();
      failed = runHadFailure(result);
      wrap.classList.add("live-code--ran");
      if (resetBtn) {
        resetBtn.disabled = false;
      }
      if (!wrap.classList.contains("live-code--editing")) {
        wrap.dataset.liveOriginal = code;
      }
    } catch (error) {
      failed = true;
      const message = error instanceof Error ? error.message : String(error);
      void renderConsole(consoleEl, { error: message });
    } finally {
      setRunFailed(wrap, failed);
      runBtn.disabled = false;
    }
  }
}

function unwrapAll() {
  document.querySelectorAll(".live-code").forEach((wrap) => {
    void destroyEditor(wrap);
    const block = wrap.querySelector(".live-code__block");
    const highlightRoot = block?.querySelector(".highlight, pre");
    if (highlightRoot?.parentNode && wrap.parentNode) {
      wrap.parentNode.insertBefore(highlightRoot, wrap);
    }
    wrap.remove();
  });
}

function wrapAllBlocks() {
  for (const codeEl of findRunnableBlocks()) {
    const wrap = wrapBlock(codeEl);
    if (wrap) {
      wrap.dataset.liveBaseline = wrap.dataset.liveOriginal ?? "";
    }
  }
  document.querySelectorAll(".live-code").forEach((wrap) => {
    stripMaterialCopyButtons(wrap);
  });
}

function ensureLaunchBar() {
  let bar = document.getElementById("live-code-bar");
  if (bar) {
    return bar;
  }

  bar = document.createElement("div");
  bar.id = "live-code-bar";
  bar.className = "live-code-bar live-code-bar--inactive";
  bar.setAttribute("role", "status");
  bar.innerHTML = [
    '<span id="live-code-status" class="live-code-bar__status">Inactive</span>',
    '<button type="button" id="live-code-toggle" class="live-code-bar__toggle"',
    ' aria-label="Turn on live code" aria-pressed="false">',
    '<span class="live-code-bar__dot" aria-hidden="true"></span>',
    "</button>",
  ].join("");
  document.body.appendChild(bar);
  return bar;
}

function updateBarUi() {
  const bar = document.getElementById("live-code-bar");
  const status = document.getElementById("live-code-status");
  const toggle = document.getElementById("live-code-toggle");
  if (!bar || !status || !toggle) {
    return;
  }

  bar.className = `live-code-bar live-code-bar--${connectionState}`;

  const labels = {
    uninstalled: "Uninstalled",
    inactive: "Inactive",
    live: "Live",
  };
  status.textContent = labels[connectionState];

  const isOn = connectionState === "live";
  toggle.setAttribute("aria-pressed", isOn ? "true" : "false");
  toggle.setAttribute(
    "aria-label",
    connectionState === "uninstalled"
      ? "Retry Jupyter connection"
      : isOn
        ? "Turn off live code"
        : "Turn on live code",
  );
}

async function connectLive() {
  const status = document.getElementById("live-code-status");
  if (status) {
    status.textContent = "Connecting…";
  }
  try {
    await getKernel();
    connectionState = "live";
    sessionStorage.setItem(STORAGE_KEY, "1");
  } catch (error) {
    console.error(error);
    await resetKernel();
    sessionStorage.removeItem(STORAGE_KEY);
    connectionState = (await probeJupyter()) ? "inactive" : "uninstalled";
  } finally {
    updateBarUi();
  }
}

function disconnectLive() {
  connectionState = "inactive";
  void resetKernel();
  sessionStorage.removeItem(STORAGE_KEY);
  updateBarUi();
}

async function refreshInstallState() {
  const status = document.getElementById("live-code-status");
  if (status) {
    status.textContent = "Checking…";
  }

  const available = await probeJupyter();
  if (!available) {
    connectionState = "uninstalled";
    void resetKernel();
    sessionStorage.removeItem(STORAGE_KEY);
  } else if (connectionState === "uninstalled") {
    connectionState = "inactive";
  }
  updateBarUi();
  return available;
}

async function onToggleClick() {
  if (connectionState === "uninstalled") {
    const available = await refreshInstallState();
    if (!available) {
      return;
    }
  }

  const toggle = document.getElementById("live-code-toggle");
  const isOn = toggle?.getAttribute("aria-pressed") === "true";
  if (isOn) {
    disconnectLive();
    return;
  }

  await connectLive();
}

function wireLaunchBar() {
  const toggle = document.getElementById("live-code-toggle");
  if (!toggle || toggle.dataset.liveWired === "true") {
    return;
  }
  toggle.dataset.liveWired = "true";
  toggle.addEventListener("click", () => {
    void onToggleClick();
  });
}

async function initPage() {
  unwrapAll();
  ensureLaunchBar();
  wireLaunchBar();
  wrapAllBlocks();

  const wantsLive = sessionStorage.getItem(STORAGE_KEY) === "1";
  if (!wantsLive && connectionState === "live") {
    const stale = activeKernel;
    connectionState = "inactive";
    activeKernel = null;
    kernelFlight = null;
    void disposeKernel(stale);
  }

  await refreshInstallState();

  if (wantsLive && connectionState === "inactive") {
    activeKernel = null;
    kernelFlight = null;
    await connectLive();
  } else {
    updateBarUi();
  }
}

function scheduleInit() {
  let started = false;
  const run = () => {
    if (started) {
      return;
    }
    started = true;
    void initPage();
  };
  document.body.addEventListener("docs-shiki-ready", run, { once: true });
  setTimeout(run, 2500);
}

window.addEventListener("focus", () => {
  if (connectionState === "uninstalled") {
    void refreshInstallState();
  }
});

if (typeof document$ !== "undefined") {
  document$.subscribe(scheduleInit);
} else {
  document.addEventListener("DOMContentLoaded", scheduleInit);
}
